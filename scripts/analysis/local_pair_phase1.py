#!/usr/bin/env python3
"""Summarize the local mixed-GPU probe artifacts without pooling instruments."""

import argparse
import csv
from datetime import datetime
import json
import statistics
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def metrics(path):
    if not path.exists():
        return {}
    return {parts[0]: float(parts[1]) for line in path.read_text().splitlines()
            if not line.startswith('#') and len(parts := line.split()) == 2}


def median(records, field):
    values = [row[field] for row in records if row.get(field) is not None]
    return round(statistics.median(values), 2) if values else None


def summarize(arm):
    shallow = rows(arm / 'engines.jsonl')
    deep = rows(arm / 'depth.jsonl')
    props = json.loads((arm / 'props.json').read_text()) if (arm / 'props.json').exists() else {}
    concurrent_path = arm / 'concurrency/summary.json'
    concurrent = json.loads(concurrent_path.read_text()) if concurrent_path.exists() else {}
    before = metrics(arm / 'metrics-before.txt')
    after = metrics(arm / 'metrics-shallow.txt')
    delta = {key: value - before.get(key, 0) for key, value in after.items()}
    devices = {}
    if (arm / 'gpu.csv').exists():
        with (arm / 'gpu.csv').open() as stream:
            for raw in csv.DictReader(stream, skipinitialspace=True):
                row = {k.strip(): v.strip() for k, v in raw.items() if k and v}
                try:
                    index = row['index']
                    used = float(row['memory.used [MiB]'].split()[0])
                    free = float(row['memory.free [MiB]'].split()[0])
                    temperature = float(row['temperature.gpu'])
                except (KeyError, ValueError):
                    continue
                device = devices.setdefault(index, {
                    'name': row['name'], 'uuid': row['uuid'],
                    'initial_used_mib': used, 'max_used_mib': used, 'min_free_mib': free,
                    'max_temperature_c': temperature, 'max_link_gen': 0, 'link_widths': [],
                    'initial_temperature_c': temperature, 'final_temperature_c': temperature,
                    'max_power_w': 0, 'busy_power_w': [], 'max_fan_percent': None,
                    'samples': [],
                })
                device['max_used_mib'] = max(device['max_used_mib'], used)
                device['min_free_mib'] = min(device['min_free_mib'], free)
                device['max_temperature_c'] = max(device['max_temperature_c'], temperature)
                device['final_temperature_c'] = temperature
                power = float(row['power.draw [W]'].split()[0])
                device['samples'].append((datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f'),
                                          power, temperature))
                device['max_power_w'] = max(device['max_power_w'], power)
                if float(row['utilization.gpu [%]'].split()[0]) >= 50:
                    device['busy_power_w'].append(power)
                if 'fan.speed [%]' in row:
                    device['max_fan_percent'] = max(device['max_fan_percent'] or 0,
                                                   float(row['fan.speed [%]'].split()[0]))
                device['max_link_gen'] = max(device['max_link_gen'], int(row['pcie.link.gen.current']))
                width = int(row['pcie.link.width.current'])
                if width not in device['link_widths']:
                    device['link_widths'].append(width)
    for device in devices.values():
        busy = device.pop('busy_power_w')
        device['busy_power_median_w'] = round(statistics.median(busy), 2) if busy else None
        samples = device.pop('samples')
        device['observed_seconds'] = round((samples[-1][0] - samples[0][0]).total_seconds(), 2)
        device['sampled_energy_wh'] = round(sum(
            (right[0] - left[0]).total_seconds() * (left[1] + right[1]) / 2
            for left, right in zip(samples, samples[1:])) / 3600, 3)
        buckets = {}
        for timestamp, power, temperature in samples:
            minute = int((timestamp - samples[0][0]).total_seconds() // 60)
            bucket = buckets.setdefault(minute, {'temperature_c': [], 'power_w': []})
            bucket['temperature_c'].append(temperature)
            bucket['power_w'].append(power)
        device['minute_medians'] = {
            minute: {key: round(statistics.median(values), 2) for key, values in bucket.items()}
            for minute, bucket in buckets.items()}
    drafted = delta.get('llamacpp:spec_decode_num_draft_tokens_total', 0)
    accepted = delta.get('llamacpp:spec_decode_num_accepted_tokens_total', 0)
    prompt_seconds = delta.get('llamacpp:prompt_seconds_total', 0)
    return {
        'arm': arm.name,
        'context': props.get('default_generation_settings', {}).get('n_ctx'),
        'slots': props.get('total_slots'),
        'engine_commit': props.get('build_info'),
        'shallow_rows': len(shallow),
        'errors': [row['error'] for row in shallow + deep if row.get('error')],
        'shallow_decode_median': median(shallow, 'decode_tok_s'),
        'shallow_prefill_server': round(delta.get('llamacpp:prompt_tokens_total', 0) / prompt_seconds, 2)
        if prompt_seconds else None,
        'shallow_acceptance': accepted / drafted if drafted else None,
        'depth': [{key: row.get(key) for key in (
            'depth_target', 'depth_padding_tokens', 'server_high_water_tokens',
            'decode_tps', 'prefill_tps', 'draft_accept_rate', 'error')}
            for row in deep],
        'devices': devices,
        'concurrency': {key: concurrent.get(key) for key in (
            'complete', 'mtp_live', 'requested_requests', 'padding_tokens', 'makespan_seconds',
            'completion_tokens', 'output_tokens_per_wall_second',
            'output_tokens_per_first_token_to_last_finish_second',
            'max_observed_busy_slots', 'mtp', 'requests')} if concurrent else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--expect-rows', type=int,
                        help='validate one finished arm instead of summarizing the experiment')
    args = parser.parse_args()
    if args.expect_rows is not None:
        measured = rows(args.root / 'engines.jsonl') + rows(args.root / 'depth.jsonl')
        if len(measured) != args.expect_rows or any(row.get('error') for row in measured):
            raise SystemExit('probe has missing or failed measurements; inspect JSONL')
        return
    results = [summarize(arm) for arm in sorted(args.root.iterdir())
               if arm.is_dir() and (arm / 'server.log').exists()]
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

"""Summarize exact task counters and rounded server intervals without inventing token regions."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

ROOT = Path('/home/xanewok/repos/qwen38-eval/results/2026-09-28-dual-5060ti')
PAT = re.compile(r'(\d\d-\d\d \d\d:\d\d:\d\d).*Accepted: (\d+) tokens, Drafted: (\d+) tokens, Per-position acceptance rate: ([\d.]+(?:, [\d.]+){4})')


def call_class(c):
    ratio = c.get('reasoning_tokens', 0) / max(c.get('output_tokens', 0), 1)
    return 'reasoning-heavy-call' if ratio >= .8 else 'reasoning-light-call' if ratio <= .2 else 'mixed-call'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--out', type=Path, default=Path('codex/campaigns.json'))
    args = p.parse_args()
    rows, intervals, launches, log_features = [], [], {}, {}
    for campaign in sorted(args.root.glob('campaign-hq15-ckptab-*-20261001')):
        tag = campaign.name.removeprefix('campaign-hq15-ckptab-').removesuffix('-20261001')
        calls = []
        for job in sorted(campaign.glob('job-*.json')):
            j = json.loads(job.read_text()); attempt = campaign / j['attempt']
            m = j['probe']['metrics_delta']
            cs = [json.loads(s) for s in (attempt/'model-calls.jsonl').read_text().splitlines() if s]
            calls.extend(cs)
            row = dict(arm=tag, task=j['task'], file=str(job), status=j['status'],
                       accepted=m['draft_accepted_tokens'], drafted=m['draft_tokens'], rounds=m['drafts'],
                       decode_tokens=m['decode_tokens'], decode_seconds=m['decode_seconds'],
                       acceptance_D=m['draft_accepted_tokens']/m['draft_tokens'],
                       calls_M=len(cs), reasoning_M=sum(c.get('reasoning_tokens', 0) for c in cs),
                       output_M=sum(c.get('output_tokens', 0) for c in cs))
            row['reasoning_share_D'] = row['reasoning_M']/row['output_M']
            rows.append(row)
            launch = json.loads((attempt/'launch.json').read_text())
            cmd = launch['command'][:]
            cmd[3] = '<model>'
            launches.setdefault(tag, []).append(dict(command=cmd, env=launch['env'],
                                                        template=launch['chat_template_sha256'], version=launch['vllm_version']))
        log = (campaign/'serve.log').read_text()
        log_features[tag] = dict(draft_heads=sorted(set(re.findall(r'MTP drafter uses a (\d+)-token draft head',log))),
                                 blocks=sorted(set(re.findall(r'Setting attention block size to (\d+) tokens',log))),
                                 warnings=sorted(set(re.sub(r'^.*?\[[^\]]+\] ', '', s) for s in log.splitlines() if ' WARNING ' in s)))
        for line_number, line in enumerate(log.splitlines(), 1):
            match = PAT.search(line)
            if not match:
                continue
            stamp, accepted, drafted, positions = match.groups()
            end = datetime.strptime('2026-'+stamp, '%Y-%m-%d %H:%M:%S').replace(tzinfo=ZoneInfo('Europe/Warsaw')).timestamp()
            # Only complete, nominal ten-second intervals inside one request are classified.
            start = end - 10
            contained = [c for c in calls if c['started'] <= start and c['finished'] >= end]
            label = call_class(contained[0]) if len(contained) == 1 else 'unassigned'
            intervals.append(dict(arm=tag, line=line_number, start=start, end=end, label=label,
                                  accepted=int(accepted), drafted=int(drafted),
                                  positions=[float(x) for x in positions.split(',')]))
    sums = {}
    for arm in sorted({r['arm'] for r in rows}):
        rs=[r for r in rows if r['arm']==arm]
        acc=sum(r['accepted'] for r in rs); draft=sum(r['drafted'] for r in rs)
        dec=sum(r['decode_tokens'] for r in rs); sec=sum(r['decode_seconds'] for r in rs)
        sums[arm]=dict(accepted_M=acc,drafted_M=draft,acceptance_D=acc/draft,
                       decode_tps_D=dec/sec,task_mean_acceptance_D=sum(r['acceptance_D'] for r in rs)/len(rs),
                       reasoning_share_D=sum(r['reasoning_M'] for r in rs)/sum(r['output_M'] for r in rs))
    bins = {}
    for arm in sums:
        bins[arm]={}
        for label in ('reasoning-heavy-call','reasoning-light-call','mixed-call','unassigned'):
            xs=[r for r in intervals if r['arm']==arm and r['label']==label]
            drafted=sum(r['drafted'] for r in xs)
            if drafted:
                bins[arm][label]=dict(windows_M=len(xs),drafted_M=drafted,
                                     acceptance_D=sum(r['accepted'] for r in xs)/drafted,
                                     position_survival_D=[sum(r['positions'][i]*r['drafted'] for r in xs)/drafted for i in range(5)])
    same_launch = all(v == next(iter(launches.values()))[0] for vs in launches.values() for v in vs)
    args.out.write_text(json.dumps(dict(rows=rows,summary=sums,call_proxy_bins=bins,intervals=intervals,
                                        identical_launch_except_model=same_launch,log_features=log_features),indent=2)+'\n')
    print('arm\taccepted_M\tdrafted_M\tacceptance_D\ttask_mean_D\tdecode_tps_D\treasoning_share_D')
    for arm,s in sums.items():
        print(f"{arm}\t{s['accepted_M']:.0f}\t{s['drafted_M']:.0f}\t{s['acceptance_D']:.6f}\t{s['task_mean_acceptance_D']:.6f}\t{s['decode_tps_D']:.3f}\t{s['reasoning_share_D']:.6f}")
    print('\ntask\tg32i8_r1_D\tg32i8_r2_D\tcodexh_r1_D\tcodexh_r2_D\tdelta_pp_D')
    for task in sorted({r['task'] for r in rows}):
        rates={r['arm']:r['acceptance_D'] for r in rows if r['task']==task}
        vals=[rates[a] for a in ('g32i8-r1','g32i8-r2','codexh-r1','codexh-r2')]
        print(task+'\t'+'\t'.join(f'{v*100:.2f}' for v in vals)+f'\t{(sum(vals[2:])-sum(vals[:2]))*50:.2f}')
    print('\nidentical_launch_except_model',same_launch)


if __name__ == '__main__':
    main()

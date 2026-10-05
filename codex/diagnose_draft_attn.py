#!/usr/bin/env python3
"""Read recorded kernel resources; derive unique-KV effective bandwidth."""
import json
from pathlib import Path
import sqlite3

root = Path('/home/xanewok/repos/attrib-pack')
result = {'labels': {'M': 'recorded', 'D': 'arithmetic', 'G': 'unmeasured candidate'}}
for tag, kv in (('8k', 8192), ('90k', 90112)):
    path = root / f'runs/attrib-host-s3m/full-a1/decode-{tag}.sqlite'
    db = sqlite3.connect(f'file:{path}?mode=ro&immutable=1', uri=True)
    query = '''select s.value,k.deviceId,k.gridX,k.blockX,k.blockY,k.registersPerThread,
        k.dynamicSharedMemory,count(*),avg(k.end-k.start)/1000.0,
        min(k.end-k.start)/1000.0,max(k.end-k.start)/1000.0
        from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName
        where s.value like 'void flash_attn_ext_f16<(int)256, (int)256, (int)1, (int)8,%'
        group by s.value,k.deviceId,k.gridX,k.blockX,k.blockY,k.registersPerThread,k.dynamicSharedMemory'''
    nbytes = kv * 2 * 256 * 2 * 2
    rows = []
    for name, gpu, grid, bx, by, regs, shared, count, us, minimum, maximum in db.execute(query):
        rows.append({'M_gpu': gpu, 'M_grid_x': grid, 'M_threads': bx * by, 'M_registers_per_thread': regs,
                     'M_shared_bytes': shared, 'M_samples': count, 'M_us': us, 'M_range_us': [minimum, maximum],
                     'D_kv_bytes': nbytes, 'D_effective_gbs': nbytes / us / 1000,
                     'D_fraction_of_448_gbs': nbytes / us / 1000 / 448, 'kernel': name})
    result[tag] = {'trace': str(path), 'rows': rows}
    db.close()
device_db = root / 'runs/attrib-host-s3m/full-a1/decode-90k.sqlite'
db = sqlite3.connect(f'file:{device_db}?mode=ro&immutable=1', uri=True)
db.row_factory = sqlite3.Row
devices = [dict(row) for row in db.execute('select id,smCount,maxRegistersPerSm,maxShmemPerSm,maxShmemPerBlockOptin,threadsPerWarp,maxWarpsPerSm from TARGET_INFO_GPU')]
db.close()
ab_path = Path('/home/xanewok/repos/stack3-rebase/codex/runs/draft-kv-q8-ab/full-a1/summary.json')
ab = json.loads(ab_path.read_text())
result['q8_0'] = {
    'M_device_trace': str(device_db), 'M_devices': devices,
    'M_round_source': str(ab_path),
    'D_base_kernel': 'flash_attn_ext_vec<256,1,Q8_0,Q8_0,false>',
    'D_base_grid': [1, 6, 12], 'D_base_block': [32, 4, 1],
    'D_patch_kernel': 'flash_attn_ext_f16<256,256,1,8,false,false,false,Q8_0>',
    'D_patch_grid': [36, 1, 1], 'D_patch_block': [32, 4, 1],
    'D_patch_dynamic_shared_bytes': 83088,
    'note': 'Launch geometry derives from compile resources and recorded device limits; no q8_0 one-row kernel trace found. Payload counts exclude repeated scale loads and are not DRAM counters.',
    'depths': [],
}
for kv in (8192, 32768, 90112):
    unique = kv * 2 * (256 // 32 * 34) * 2
    measured = ab[str(kv)]
    result['q8_0']['depths'].append({
        'kv': kv, 'D_unique_q8_bytes': unique, 'D_base_payload_scanned_bytes': 6*unique,
        'D_f16_unique_bytes': kv*2*256*2*2,
        'D_base_max_iterations': (kv + 6*128 - 1)//(6*128),
        'D_patch_max_iterations': ((kv//64) + 18 - 1)//18,
        'D_peak_floor_us': unique / 448032,
        'G_five_pass_ms': 5*(unique / 418000 + 12) / 1000,
        'D_recorded_round_delta_ms': measured['patch_minus_up_ms_D'],
        'M_up_acceptance': measured['up_acceptance_M'],
        'M_q8_acceptance': measured['patch_acceptance_M'],
    })
print(json.dumps(result, indent=2))

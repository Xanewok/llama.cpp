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
print(json.dumps(result, indent=2))

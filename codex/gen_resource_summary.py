"""Extract ptxas measurements and derive the dynamic SRAM requirement of each small-J config."""
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
logs = [(p.name.split('-', 1)[0], p) for p in (HERE / 'resources').glob('*.log')]
records = []
for mode, path in logs:
    name = None
    spills = None
    for line in path.read_text(errors='replace').splitlines():
        match = re.search(r"Compiling entry function '([^']+)'", line)
        if match:
            name = match.group(1)
        match = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', line)
        if match:
            spills = dict(stack_bytes_M=int(match[1]), spill_store_bytes_M=int(match[2]), spill_load_bytes_M=int(match[3]))
        match = re.search(r'Used (\d+) registers', line)
        if match and name and name.startswith('_Z9mul_mat_qI'):
            decoded = subprocess.check_output(['c++filt', name], text=True).strip()
            signature = re.search(r'mul_mat_q<\(ggml_type\)(\d+), (\d+), (false|true)', decoded)
            if not signature or int(signature[2]) not in (8, 16, 24):
                continue
            J = int(signature[2])
            if mode == 'base' and J == 24 and signature[3] == 'true':
                continue
            I, threads = (64, 128) if mode == 'patch' else (128, 256)
            y_bytes = (J*144 + threads*4-1) // (threads*4) * (threads*4)
            shared = re.search(r'(\d+) bytes smem', line)
            records.append(dict(mode=mode, signature=decoded.split('>(')[0] + '>',
                                type=int(signature[1]), J=J, fallback=signature[3] == 'true',
                                registers_M=int(match[1]), static_smem_bytes_M=int(shared[1]) if shared else 0,
                                dynamic_smem_bytes_D=I*76*4 + y_bytes + J*4, **(spills or {})))
assert records, 'no ptxas records'
(HERE / 'resources/summary.json').write_text(json.dumps(records, indent=2))
for row in records:
    print(row['mode'], row['type'], row['J'], row['fallback'], row['registers_M'], row['static_smem_bytes_M'],
          row['dynamic_smem_bytes_D'], row.get('spill_store_bytes_M'), row.get('spill_load_bytes_M'))

"""Check CUDA compiler declarations; this never executes CUDA code."""
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'build-audit'
old = (out / 'v4-q8.ptx').read_text()
new = (out / 'v5-q8.ptx').read_text()
pattern = r'\.shared \.align (\d+) \.b8 (\S*Li256ELi1\S*E2KQ)\['
old_align = re.findall(pattern, old)
new_align = re.findall(pattern, new)
assert old_align and all(int(a) == 4 for a, _ in old_align)
assert new_align and all(int(a) == 16 for a, _ in new_align)
assert 'st.shared.v2.b32' in old
for filename, names in [('v5-tile.ptx', ('Q_tmp', 'KV_tmp', 'KQ')), ('v5-mma.ptx', ('tile_Q',))]:
    text = (out / filename).read_text()
    for name in names:
        decls = re.findall(r'\.shared \.align (\d+) \.b8 (\S*' + name + r')\[', text)
        assert decls and all(int(a) >= 16 for a, _ in decls), (filename, name, decls)
for tag in ('v4', 'v5'):
    symbols = subprocess.check_output(['readelf', '-sW', str(out / f'{tag}-q8-debug.cubin')], text=True)
    (out / f'{tag}-debug-symbols.txt').write_text(symbols)
    offsets = [int(m[0], 16) for m in re.findall(r'^\s*\d+:\s+([0-9a-f]+).* (\S*Li256ELi1\S*E2KQ)$', symbols, re.M)]
    assert offsets
    if tag == 'v4':
        assert all((address + 256) % 8 == 4 for address in offsets)
    else:
        assert all(address % 16 == 0 for address in offsets)
    print(f'M: {tag} compiler shared KQ offsets={[hex(address) for address in offsets]}')
print('M: VEC/TILE/MMA shared alignment declarations pass; v4 float2 store control rejected')

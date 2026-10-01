"""Compare first-slot probability overlap and final hidden states across interleaved boots."""
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import torch

torch.set_num_threads(8)
root=Path(sys.argv[1])
rows=defaultdict(list)
hidden=defaultdict(list)
paths=sorted(root.glob('[0-9]-*.json'))
if [p.stem for p in paths] != ['0-up','1-patch','2-patch','3-up']:
    raise ValueError('incomplete paired boots')
for path in paths:
    arm=path.stem.split('-',1)[1]
    data=json.loads(path.read_text())
    if not data['rows']:
        raise ValueError('empty probe results')
    hs=torch.load(path.with_suffix('.hidden.pt'),map_location='cpu',weights_only=True)
    for row in data['rows']:
        name=row['case']
        hidden[arm,name].append(hs[name].float())
        for filt,v in row['overlap_D'].items():
            rows[arm,name,filt].append(torch.tensor(v))
print('case\tfilter\tup_overlap_D\tpatch_overlap_D\tdelta_pp_D\tup_repeat_max_D\tpatch_repeat_max_D')
for name,filt in sorted({(k[1],k[2]) for k in rows}):
    u=torch.stack(rows['up',name,filt]); p=torch.stack(rows['patch',name,filt])
    if u.shape != p.shape or len(u) != 2:
        raise ValueError('incomplete paired boots')
    if os.environ.get('CONTROL_IDENTICAL') == '1':
        torch.testing.assert_close(u,p,atol=1e-6,rtol=1e-6)
    print(f'{name}\t{filt}\t{u.mean():.6f}\t{p.mean():.6f}\t{100*(p-u).mean():.4f}\t{(u[0]-u[1]).abs().max():.6f}\t{(p[0]-p[1]).abs().max():.6f}')
for name in sorted({k[1] for k in hidden}):
    u,p=hidden['up',name][0],hidden['patch',name][0]
    if os.environ.get('CONTROL_IDENTICAL') == '1':
        torch.testing.assert_close(u,p,atol=1e-6,rtol=1e-6)
    cosine=torch.nn.functional.cosine_similarity(u,p,dim=-1)
    relative=(p-u).norm()/u.norm()
    print(f'{name}\thidden_cosine_mean_D={cosine.mean():.6f}\thidden_relative_l2_D={relative:.6f}')

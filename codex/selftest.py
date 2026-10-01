"""Check audit sensitivity using a changed MTP tensor, without loading a model."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import torch
from safetensors.torch import save_file
from checkpoints import audit, compare

start=time.perf_counter()
with TemporaryDirectory(dir=Path(__file__).parent,prefix='build-test-') as tmp:
    root=Path(tmp)
    paths=[root/'up',root/'patch']
    for path in paths:
        path.mkdir()
        save_file({'mtp.fc.weight':torch.tensor([[1.,2.]],dtype=torch.bfloat16)},path/'model.safetensors')
        (path/'config.json').write_text('{}')
    a,b=[audit(path) for path in paths]
    assert compare(a,b)['changed']==[]
    expected=hashlib.sha256(torch.tensor([[1.,2.]],dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes()).hexdigest()
    assert a['tensors']['mtp.fc.weight']['sha256']==expected
    save_file({'mtp.fc.weight':torch.tensor([[1.,3.]],dtype=torch.bfloat16)},paths[1]/'model.safetensors')
    assert compare(a,audit(paths[1]))['changed']==['mtp.fc.weight'], 'negative control missed changed MTP'
    (paths[1]/'config.json').write_text('{"changed":true}')
    assert compare(a,audit(paths[1]))['file_diffs']==['config.json']
print(f'CPU audit negative controls PASS; wall_seconds_M={time.perf_counter()-start:.6f}')

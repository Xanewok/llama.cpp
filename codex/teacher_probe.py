"""Stream dequantized layers to measure first-draft overlap on fixed teacher tokens.

This isolates body weights with unquantized KV. It is not a vLLM performance test
or an estimate of multi-step acceptance along sampled draft trajectories.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='8')
import torch
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding,
)

torch.set_num_threads(8)


class Weights:
    def __init__(self, root, device, dtype):
        self.root, self.device, self.dtype = Path(root), device, dtype
        self.index = json.loads((self.root/'model.safetensors.index.json').read_text())['weight_map']

    def tensor(self, name, rows=None):
        with safe_open(self.root/self.index[name], framework='pt', device='cpu') as f:
            x = f.get_tensor(name)
            if rows is not None:
                x = x[rows]
            return x.to(self.device)

    def weight(self, name, rows=None):
        if name+'.weight' in self.index:
            return self.tensor(name+'.weight', rows).to(self.dtype)
        packed = self.tensor(name+'.weight_packed', rows)
        shape = self.tensor(name+'.weight_shape').tolist()
        bits = 32 * packed.shape[1] // shape[1]
        if bits not in (4, 8) or packed.shape[1] * (32 // bits) != shape[1]:
            raise ValueError(f'unsupported packing: {name}')
        if name+'.weight_g_idx' in self.index or name+'.weight_zero_point' in self.index:
            raise ValueError(f'unsupported non-static symmetric groups: {name}')
        scale = self.tensor(name+'.weight_scale', rows).float()
        shifts = torch.arange(32 // bits, device=self.device) * bits
        q = (((packed.unsqueeze(-1) >> shifts) & ((1 << bits)-1)) - (1 << (bits-1)))
        q = q.flatten(1).float()
        group = shape[1] // scale.shape[1]
        return (q.reshape(q.shape[0], -1, group) * scale.unsqueeze(-1)).flatten(1).to(self.dtype)

    def layer(self, config, i, prefix):
        with torch.device('meta'):
            layer = Qwen3_5DecoderLayer(config, i)
        state = {}
        for key in layer.state_dict():
            name = prefix+'.'+key
            state[key] = self.weight(name.removesuffix('.weight')) if key.endswith('.weight') else self.tensor(name).to(self.dtype)
        layer.load_state_dict(state, strict=True, assign=True)
        return layer.eval()

    def norm(self, name, x, eps):
        return rms(x, self.weight(name), eps)

    def head(self, name, hidden):
        rows = int(self.tensor(name+'.weight_shape')[0])
        result = []
        for start in range(0, rows, 4096):
            w = self.weight(name, slice(start, min(start+4096, rows)))
            result.append(torch.nn.functional.linear(hidden, w).float())
        return torch.cat(result, dim=-1)


def rms(x, weight, eps):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (y * (1 + weight.float())).to(x.dtype)


def probabilities(logits, filtered):
    x = logits.float().clone()
    if filtered:
        # vLLM applies min_p before top_k/top_p. No request-specific processors are modeled.
        x.masked_fill_(x < x.amax(-1, keepdim=True) + torch.log(x.new_tensor(.05)), -torch.inf)
        threshold = x.topk(min(20, x.shape[-1]), dim=-1).values[:, -1:]
        x.masked_fill_(x < threshold, -torch.inf)
        values, ids = x.sort(dim=-1, descending=True)
        mass = values.softmax(-1).cumsum(-1)
        remove = mass - values.softmax(-1) >= .95
        values.masked_fill_(remove, -torch.inf)
        x.fill_(-torch.inf).scatter_(1, ids, values)
    return x.softmax(-1)


def overlap(p, q):
    return torch.minimum(p, q).sum(-1)


def body(w, cfg, ids):
    hidden = w.weight('model.language_model.embed_tokens', ids[0]).unsqueeze(0)
    pos = torch.arange(ids.shape[1], device=w.device).unsqueeze(0)
    rotary = Qwen3_5TextRotaryEmbedding(cfg).to(w.device)
    pe = rotary(hidden, pos.unsqueeze(0).expand(3, -1, -1))
    mask = torch.full((ids.shape[1], ids.shape[1]), -torch.inf, device=w.device, dtype=w.dtype).triu(1)[None,None]
    for i in range(cfg.num_hidden_layers):
        layer = w.layer(cfg, i, f'model.language_model.layers.{i}')
        hidden = layer(hidden, position_embeddings=pe, position_ids=pos,
                       attention_mask=mask if cfg.layer_types[i] == 'full_attention' else None,
                       use_cache=False)
        del layer
    return w.norm('model.language_model.norm', hidden, cfg.rms_norm_eps)


def draft(w, cfg, ids, hidden):
    # h[t] and embedding x[t+1] produce q(x[t+2]); p comes from h[t+1].
    emb = w.weight('model.language_model.embed_tokens', ids[0,1:]).unsqueeze(0)
    mixed = torch.cat([w.norm('mtp.pre_fc_norm_embedding', emb, cfg.rms_norm_eps),
                       w.norm('mtp.pre_fc_norm_hidden', hidden[:,:-1], cfg.rms_norm_eps)], -1)
    mixed = torch.nn.functional.linear(mixed, w.weight('mtp.fc'))
    mtp_cfg = copy.deepcopy(cfg)
    mtp_cfg.layer_types = ['full_attention']
    mtp_cfg.num_hidden_layers = 1
    layer = w.layer(mtp_cfg, 0, 'mtp.layers.0')
    pos = torch.arange(1, ids.shape[1], device=w.device).unsqueeze(0)
    pe = Qwen3_5TextRotaryEmbedding(mtp_cfg).to(w.device)(mixed, pos.unsqueeze(0).expand(3,-1,-1))
    mask = torch.full((pos.shape[1],pos.shape[1]), -torch.inf, device=w.device,dtype=w.dtype).triu(1)[None,None]
    out = layer(mixed, position_embeddings=pe, position_ids=pos, attention_mask=mask, use_cache=False)
    return w.norm('mtp.norm', out, cfg.rms_norm_eps)


def self_test():
    torch.manual_seed(19)
    # Compare streamed full/linear attention masks with the canonical HF model on CPU.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    cfg = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                          head_dim=8, linear_num_key_heads=1, linear_num_value_heads=2,
                          linear_key_head_dim=8, linear_value_head_dim=8,
                          layer_types=['linear_attention','full_attention'],
                          rope_parameters={'rope_type':'default','rope_theta':10000.,
                                           'partial_rotary_factor':1.,'mrope_section':[1,1,2]})
    cfg._attn_implementation = 'eager'
    model = Qwen3_5TextModel(cfg).eval()
    ids = torch.tensor([[1,2,3,4]])
    class Tiny:
        device, dtype = 'cpu', torch.float32
        def weight(self, name, rows=None):
            x = model.state_dict()[name.removeprefix('model.language_model.')+'.weight']
            return x if rows is None else x[rows]
        def layer(self, config, i, prefix):
            return model.layers[i]
        def norm(self, name, x, eps):
            return rms(x, self.weight(name), eps)
    expected = model(ids, use_cache=False).last_hidden_state
    got = body(Tiny(), cfg, ids)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)
    p, q = torch.tensor([[.8,.2]]), torch.tensor([[.4,.6]])
    torch.testing.assert_close(overlap(p,q), torch.tensor([.6]))
    pp = probabilities(torch.randn(3,32), True)
    torch.testing.assert_close(overlap(pp,pp), torch.ones(3))
    bad = overlap(pp, pp.roll(1,-1))
    assert (bad < .99).any(), 'negative control did not detect a changed draft distribution'
    # The packed sign offset must survive dequantization; reversing it is the negative control.
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32, unpack_from_int32
    for bits in (4,8):
        vals = torch.arange(-(1<<(bits-1)), 1<<(bits-1), dtype=torch.int8).reshape(1,-1)
        packed = pack_to_int32(vals,bits)
        shifts = torch.arange(32//bits)*bits
        got = (((packed.unsqueeze(-1)>>shifts)&((1<<bits)-1))-(1<<(bits-1))).flatten(1).to(torch.int8)
        assert torch.equal(got, unpack_from_int32(packed,bits,vals.shape))
        assert not torch.equal((got.float()+(1<<(bits-1))),vals.float())
    from tempfile import TemporaryDirectory
    from safetensors.torch import save_file
    with TemporaryDirectory(dir=Path(__file__).parent, prefix='build-test-') as directory:
        root = Path(directory)
        tensors = {}
        for bits in (4,8):
            vals = torch.arange(-8,8,dtype=torch.int8).reshape(2,8)
            prefix = f'linear{bits}'
            tensors[prefix+'.weight_packed'] = pack_to_int32(vals,bits)
            tensors[prefix+'.weight_scale'] = torch.tensor([[.5,2.],[1.,.25]])
            tensors[prefix+'.weight_shape'] = torch.tensor([2,8])
        save_file(tensors, root/'weights.safetensors')
        (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'weights.safetensors' for k in tensors}}))
        w = Weights(root,'cpu',torch.float32)
        for bits in (4,8):
            wanted = torch.arange(-8,8).reshape(2,2,4).float()*tensors[f'linear{bits}.weight_scale'].unsqueeze(-1)
            torch.testing.assert_close(w.weight(f'linear{bits}'),wanted.flatten(1))
            torch.testing.assert_close(w.weight(f'linear{bits}',slice(1,2)),wanted.flatten(1)[1:])
    print('CPU self-test: streamed HF body, overlap law, packed sign offset, negative controls PASS')


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--model', type=Path)
    p.add_argument('--tokens', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--hidden-from', type=Path, help='CPU MTP-only replay of a frozen full-body hidden dump')
    p.add_argument('--device', default='cpu', choices=['cpu','cuda:0'])
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if not all((args.model,args.tokens,args.out)):
        p.error('--model, --tokens and --out are required')
    start = time.perf_counter()
    cfg = Qwen3_5TextConfig(**json.loads((args.model/'config.json').read_text())['text_config'])
    cfg._attn_implementation = 'eager'
    w = Weights(args.model,args.device,torch.bfloat16)
    draft_ids = torch.load(args.model/'mtp_draft_vocab_ids.pt',map_location=args.device,weights_only=True).long()
    token_sha = hashlib.sha256(args.tokens.read_bytes()).hexdigest()
    cached = torch.load(args.hidden_from,map_location='cpu',weights_only=True) if args.hidden_from else None
    if cached is not None and cached['_tokens_sha256'] != token_sha:
        raise ValueError('hidden dump uses different teacher tokens')
    rows, states = [], {'_tokens_sha256':token_sha}
    for case in json.loads(args.tokens.read_text()):
        ids = torch.tensor([case['ids']],device=args.device)
        h = cached[case['name']].to(args.device).unsqueeze(0) if cached is not None else body(w,cfg,ids)
        d = draft(w,cfg,ids,h)
        # Score the continuation positions; each position has the same teacher prefix in both arms.
        offsets = torch.arange(case['score_from'],ids.shape[1],device=args.device)
        if offsets.min() < 1:
            raise ValueError('score_from must be positive')
        target = w.head('lm_head',h[0,offsets])
        sub = w.head('mtp.draft_lm_head',d[0,offsets-1])
        draft_logits = target.new_full(target.shape,-torch.inf)
        draft_logits[:,draft_ids] = sub
        row = dict(case=case['name'], positions_M=len(offsets), overlap_D={},
                   interpretation='first draft probability overlap; teacher history; BF16 KV; no grammar')
        for label,filtered in [('raw',False),('campaign_filters',True)]:
            row['overlap_D'][label] = overlap(probabilities(target,filtered),probabilities(draft_logits,filtered)).cpu().tolist()
        rows.append(row)
        states[case['name']] = h[0].cpu()
    args.out.write_text(json.dumps(dict(rows=rows,wall_seconds_M=time.perf_counter()-start),indent=2)+'\n')
    torch.save(states,args.out.with_suffix('.hidden.pt'))


if __name__ == '__main__':
    main()

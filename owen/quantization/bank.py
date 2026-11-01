"""Fixed MC-MoE GPTQ candidates. No experiment fingerprint or automatic resume."""
from collections import OrderedDict
import json
from pathlib import Path
import torch


class GPTQBank:
    def __init__(self, directory, cache_mb=512):
        self.path = Path(directory).resolve()
        self.manifest = json.loads((self.path / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('format') != 'owen-small-mcmoe-gptq-v1':
            raise ValueError('Expected a completed DeepSeek-Small MC-MoE GPTQ bank')
        self.modules = self.manifest['modules']
        self.cache, self.size, self.limit = OrderedDict(), 0, int(cache_mb * 1024 ** 2)

    def bind(self, model):
        if getattr(model, '_gptq_bank', None) is self:
            return
        modules = dict(model.named_modules())
        for name in self.modules:
            if name not in modules:
                raise ValueError(f'Candidate module missing from model: {name}')
            modules[name]._gptq_bank = self
            modules[name]._gptq_name = name
        background = torch.load(self.path / 'background.pt', map_location='cpu', weights_only=True)
        with torch.no_grad():
            for name, weight in background.items():
                module = modules[name]
                if module.weight.shape != weight.shape:
                    raise ValueError(f'Background shape mismatch: {name}')
                module.weight.copy_(weight)
        model._gptq_bank = self

    def weight(self, name, bits, module, group_size):
        if bits not in (2, 4) or group_size != self.manifest['group_size']:
            raise ValueError('Only the stored 2/4-bit candidates and group size are supported')
        key = (name, bits)
        if key in self.cache:
            tensor = self.cache.pop(key)
            self.cache[key] = tensor
        else:
            filename = self.modules[name]['candidates'][str(bits)]
            path = (self.path / filename).resolve()
            if not path.is_relative_to(self.path):
                raise ValueError('Invalid candidate path')
            tensor = torch.load(path, map_location='cpu', weights_only=True)['weight']
            size = tensor.numel() * tensor.element_size()
            while self.cache and self.size + size > self.limit:
                _, old = self.cache.popitem(last=False)
                self.size -= old.numel() * old.element_size()
            if size <= self.limit:
                self.cache[key] = tensor
                self.size += size
        if tensor.shape != module.weight.shape or not torch.isfinite(tensor).all():
            raise ValueError(f'Invalid candidate for {name}')
        return tensor.to(device=module.weight.device, dtype=module.weight.dtype, copy=True)

    def assemble(self, model, assignment):
        if set(assignment) != set(self.modules):
            raise ValueError('Assignment must cover exactly all routed candidate projections')
        if any(type(b) is not int or b not in (2, 4) for b in assignment.values()):
            raise ValueError('Assignment bits must be integer 2 or 4')
        self.bind(model)
        with torch.no_grad():
            for name, bits in assignment.items():
                module = model.get_submodule(name)
                module.weight.copy_(self.weight(name, bits, module, self.manifest['group_size']))
        return model.eval()


def bind_model(model):
    bank = getattr(model, '_gptq_bank', None)
    if bank is None:
        raise RuntimeError('Bind the fixed GPTQ bank before configuration valuation')
    bank.bind(model)


def candidate_weight(module, bits, group_size=128):
    bank = getattr(module, '_gptq_bank', None)
    if bank is None:
        raise RuntimeError('Projection is not bound to a GPTQ bank')
    return bank.weight(module._gptq_name, bits, module, group_size)

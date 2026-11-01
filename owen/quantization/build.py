"""Sequential MC-MoE GPTQ calibration, followed by immutable 2/4 candidates."""
import gc
import json
from pathlib import Path
import torch
from torch import nn
from owen.quantization.backend import GPTQ
from owen.models.model import embeddings, to_model_inputs


def candidate(module, hessian, bits, group_size=128, percdamp=.01):
    """Reference pack=True arithmetic; also retain decoded weights for switching."""
    original = module.weight.detach().clone()
    engine = GPTQ(module, None, '', bits)
    engine.H = hessian.clone()
    engine.quantizer.configure(bits, perchannel=True, sym=False, mse=False, pack=True)
    try:
        scale, zero, g_idx, error = engine.fasterquant(
            blocksize=128, percdamp=percdamp, groupsize=group_size, actorder=False)
        codes = module.weight.detach().reshape(module.weight.shape[0], -1).float()
        decoded = scale.float()[:, g_idx.long()] * (codes - zero.float()[:, g_idx.long()])
        decoded = decoded.reshape(original.shape).to(original.dtype)
        if not torch.isfinite(decoded).all():
            raise ValueError('Non-finite GPTQ candidate')
        return dict(weight=decoded.cpu(), codes=codes.to(torch.uint8).cpu(),
                    scales=scale.cpu(), zeros=zero.cpu(), g_idx=g_idx.cpu(),
                    bits=bits, error=float(error), shape=list(original.shape))
    finally:
        module.weight.data = original
        engine.free()


def collect(modules, forward, input_sources=None):
    engines = {name: GPTQ(module, None, name, 4) for name, module in modules.items()}
    handles = []
    input_sources = input_sources or {}
    for name, module in modules.items():
        if name in input_sources:
            # MLA contracts kv_b_proj.weight directly instead of invoking its
            # Linear.forward. Its input is exactly kv_a_layernorm's output.
            def hook(_module, args, output, name=name):
                engines[name].add_batch(output.detach(), output.detach())
            handles.append(input_sources[name].register_forward_hook(hook))
        else:
            def hook(_module, args, output, name=name):
                engines[name].add_batch(args[0].detach(), output.detach())
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            forward()
    finally:
        for handle in handles:
            handle.remove()
    missing = [name for name, engine in engines.items() if not engine.nsamples]
    if missing:
        raise ValueError(f'No calibration tokens reached {missing}; expand the calibration set')
    return engines


def build_bank(model, processor, samples, directory, model_path, seqlen=2048,
               group_size=128, percdamp=.01, quantize_vision=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(format='owen-small-mcmoe-gptq-v1', model_path=str(Path(model_path).resolve()),
                    dtype=str(next(model.parameters()).dtype).split('.')[-1], group_size=group_size,
                    block_size=128, damping=percdamp, act_order=False, sym=False, pack=True,
                    modules={}, budget_scope='routed_experts', seqlen=seqlen,
                    candidate_context='sequential 4-bit anchor; both candidates start from identical original weight/H',
                    background='non-router Linear/Conv2d 4-bit; embeddings/norms/router retain original precision',
                    quantize_vision=quantize_vision)
    background = {}
    counter = 0
    device = next(model.parameters()).device

    def finish(modules, engines, routed):
        nonlocal counter
        for name, module in modules.items():
            hessian = engines[name].H
            choices = (2, 4) if routed else (4,)
            for bits in choices:
                result = candidate(module, hessian, bits, group_size, percdamp)
                if routed:
                    filename = f'candidate-{counter:06d}-{bits}.pt'
                    torch.save(result, directory / filename)
                    entry = manifest['modules'].setdefault(name, dict(shape=list(module.weight.shape),
                                                                      candidates={}))
                    entry['candidates'][str(bits)] = filename
                if bits == 4:
                    module.weight.data = result['weight'].to(device=device, dtype=module.weight.dtype)
                    if not routed:
                        background[name] = result['weight']
            counter += 1
            engines[name].free()
            print(f'GPTQ prepared {name}', flush=True)
        del engines
        gc.collect()

    # Quantize the image frontend before freezing calibration embeddings.
    if quantize_vision:
        visual = [(n, m) for n, m in model.named_modules()
                  if n.startswith(('vision.', 'projector.')) and isinstance(m, (nn.Linear, nn.Conv2d))]
        for name, module in visual:
            def forward_visual():
                for sample in samples:
                    model.prepare_inputs_embeds(**to_model_inputs(model, sample))
            targets = {name: module}
            finish(targets, collect(targets, forward_visual), False)
    inputs = embeddings(model, samples, seqlen)
    torch.save(inputs, directory / 'calibration.pt')
    # Layer-sequential propagation matches the reference's quantized-input path.
    hidden = [x.to(device) for x, _ in inputs]
    for index, layer in enumerate(model.language.model.layers):
        local = dict(layer.named_modules())
        groups = [
            [n for n in local if n in ('self_attn.q_proj', 'self_attn.kv_a_proj_with_mqa')],
            ['self_attn.kv_b_proj'], ['self_attn.o_proj'],
            [n for n in local if '.experts.' not in n and n.endswith(('gate_proj', 'up_proj'))],
            [n for n in local if '.experts.' in n and n.endswith(('gate_proj', 'up_proj'))],
            [n for n in local if n.endswith('down_proj')],
        ]
        def forward_layer():
            for h in hidden:
                pos = torch.arange(h.shape[1], device=device).unsqueeze(0)
                from owen.estimation.estimator import ConfigOwenEstimator
                mask = ConfigOwenEstimator._prepare_attention_mask(model.language, h.shape[0], h.shape[1], h)
                layer(h, attention_mask=mask, position_ids=pos, use_cache=False)
        for group in groups:
            group = [n for n in group if n in local and isinstance(local[n], nn.Linear)]
            if not group:
                continue
            targets = {f'language.model.layers.{index}.{n}': local[n] for n in group}
            input_sources = {n: layer.self_attn.kv_a_layernorm for n in targets
                             if n.endswith('.self_attn.kv_b_proj')}
            engines = collect(targets, forward_layer, input_sources)
            # Split routed and fixed modules, preserving the same collected inputs.
            for routed in (False, True):
                names = [n for n in targets if ('.mlp.experts.' in n) == routed]
                if names:
                    finish({n: targets[n] for n in names}, {n: engines[n] for n in names}, routed)
        with torch.no_grad():
            from owen.estimation.estimator import ConfigOwenEstimator
            next_hidden = []
            for h in hidden:
                pos = torch.arange(h.shape[1], device=device).unsqueeze(0)
                mask = ConfigOwenEstimator._prepare_attention_mask(model.language, h.shape[0], h.shape[1], h)
                next_hidden.append(layer(h, attention_mask=mask, position_ids=pos, use_cache=False)[0].detach())
            hidden = next_hidden
    # The output head is fixed 4-bit; use final quantized decoder activations.
    head = model.language.lm_head
    def forward_head():
        for h in hidden:
            head(model.language.model.norm(h))
    targets = {'language.lm_head': head}
    finish(targets, collect(targets, forward_head), False)
    torch.save(background, directory / 'background.pt')
    if not manifest['modules']:
        raise ValueError('No routed expert candidates were generated')
    # Publish only a completely generated bank. Incomplete directories are not reused.
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest

"""Router-mass proposal and numerically stable permutation importance weights."""
import math
import torch


def permutation_weights(permutations, prior):
    """Common positive rescaling preserves the MCKP optimum, not self-normalization.

    Every sampled utility uses the SAME scale exp(-max log w). This avoids
    multiplying/clamping tiny permutation probabilities. Absolute utility units
    change by a single shared constant, while all configuration ratios remain.
    """
    p = prior.detach().double().cpu()
    if not torch.isfinite(p).all() or (p <= 0).any():
        raise ValueError('Proposal masses must be finite and strictly positive')
    logs = []
    n = len(p)
    for order in permutations.detach().cpu():
        if sorted(order.tolist()) != list(range(n)):
            raise ValueError('Each permutation must contain every expert exactly once')
        values = p[order.long()]
        suffix = values.flip(0).cumsum(0).flip(0)
        logq = (values.log() - suffix.log()).sum().item()
        logs.append(-math.lgamma(n + 1) - logq)
    if not logs:
        raise ValueError('At least one permutation is required')
    scale = max(logs)
    print(f'Importance utilities share log rescaling {-scale:.6g}', flush=True)
    return [math.exp(w - scale) for w in logs]


def profile_and_sample(model, inputs, count, seed, start_layer=1):
    """Measure top-k router weight mass, not token hit frequency."""
    layers = model.language.model.layers
    experts = layers[start_layer].mlp.config.n_routed_experts
    masses = torch.zeros((len(layers) - start_layer, experts), dtype=torch.float64)
    handles = []
    for index, layer in enumerate(layers[start_layer:]):
        def capture(_gate, _args, output, index=index):
            ids, weights = output[:2]
            ids, weights = ids.detach().cpu().reshape(-1), weights.detach().double().cpu().reshape(-1)
            masses[index].scatter_add_(0, ids.long(), weights)
        handles.append(layer.mlp.gate.register_forward_hook(capture))
    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            for embeds, _labels in inputs:
                model.language(inputs_embeds=embeds.to(device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    p = masses.flatten()
    if not torch.isfinite(p).all() or p.sum() <= 0:
        raise ValueError('No valid router mass collected')
    p = p + 1e-8
    p /= p.sum()
    generator = torch.Generator().manual_seed(seed)
    orders = torch.stack([torch.multinomial(p, len(p), replacement=False, generator=generator)
                          for _ in range(count)])
    return p, orders

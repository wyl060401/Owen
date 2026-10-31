"""Exact grouped configuration selection; budget is routed expert parameters only."""
from functools import reduce
from math import gcd, floor
import numpy as np


def solve(values, bank, average_bits):
    if not 2 <= average_bits <= 4:
        raise ValueError('Routed average bits must be between 2 and 4')
    names = set(bank.modules)
    experts = sorted({name.rsplit('.', 1)[0] for name in names})
    groups = []
    parameters = sum(int(np.prod(entry['shape'])) for entry in bank.modules.values())
    costs = []
    for expert in experts:
        layer = int(expert.split('.')[3]); eid = int(expert.split('.')[6])
        menu_values = values.get(f'L{layer}.E{eid}')
        if menu_values is None:
            raise ValueError(f'Missing configuration utilities for {expert}')
        projections = ['gate_proj', 'up_proj', 'down_proj']
        expected = {','.join(sorted(projections[j] for j in range(3) if mask & (1 << j)))
                    for mask in range(1, 8)}
        if set(menu_values) != expected:
            raise ValueError(f'Expected exactly seven nonempty configuration values for {expert}')
        items = [('', 0., 0)]
        for key, value in menu_values.items():
            if not np.isfinite(value):
                raise ValueError(f'Non-finite utility: {expert}, {key}')
            cost = sum(int(np.prod(bank.modules[expert + '.' + p]['shape'])) * 2 for p in key.split(','))
            items.append((key, float(value), cost)); costs.append(cost)
        groups.append((expert, items))
    step = reduce(gcd, costs)
    capacity = floor((average_bits - 2) * parameters + 1e-6) // step
    if len(groups) * (capacity + 1) > 100_000_000:
        raise MemoryError('Exact DP table exceeds the configured safety bound; no approximate fallback is used')
    dp = np.full(capacity + 1, -np.inf); dp[0] = 0.
    choices = np.full((len(groups), capacity + 1), -1, dtype=np.int8)
    for index, (_expert, items) in enumerate(groups):
        nxt = np.full_like(dp, -np.inf)
        for choice, (_key, value, cost) in enumerate(items):
            c = cost // step
            if c > capacity:
                continue
            candidate = dp[:capacity + 1 - c] + value
            better = candidate > nxt[c:]
            nxt[c:][better] = candidate[better]
            choices[index, c:][better] = choice
        dp = nxt
    remaining = int(np.argmax(dp)); objective = float(dp[remaining]); assignment = {}
    for index in range(len(groups) - 1, -1, -1):
        expert, items = groups[index]
        choice = int(choices[index, remaining])
        if choice < 0:
            raise RuntimeError('Invalid DP backtracking state')
        key, _, cost = items[choice]; selected = set(key.split(',')) if key else set()
        for p in ['gate_proj', 'up_proj', 'down_proj']:
            assignment[expert + '.' + p] = 4 if p in selected else 2
        remaining -= cost // step
    actual = sum(int(np.prod(bank.modules[n]['shape'])) * bit for n, bit in assignment.items()) / parameters
    if actual > average_bits + 1e-12:
        raise RuntimeError('The routed precision budget was exceeded')
    return assignment, dict(budget_scope='routed_experts', target_average_bits=average_bits,
                            actual_average_bits=actual, routed_parameters=parameters,
                            solver='exact_dp', total_utility=objective)

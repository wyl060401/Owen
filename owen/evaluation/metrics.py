"""Evaluation score export and aggregation across independent seeds."""
import csv
import json
import math
from pathlib import Path
import statistics

DEFAULT_BENCHMARKS = ('MME', 'AI2D_TEST', 'MMBench_DEV_EN_V11', 'MMStar', 'MMMU_DEV_VAL', 'OCRBench')
AVERAGE_BENCHMARKS = ('MMBench_DEV_EN_V11', 'MMStar', 'MMMU_DEV_VAL', 'AI2D_TEST', 'OCRBench')


def normalize_result(dataset, result):
    if hasattr(result, 'to_dict'):
        result = result.to_dict(orient='records')
    if isinstance(result, list):
        if len(result) != 1:
            overall = [r for r in result if str(r.get('split', '')).lower() in ('all', 'overall')]
            if len(overall) != 1:
                raise ValueError(f'{dataset}: ambiguous evaluation splits')
            result = overall
        result = result[0]
    if not isinstance(result, dict):
        raise ValueError(f'{dataset}: expected an evaluation dictionary or DataFrame')
    def number(key):
        value = result[key]
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f'{dataset}: non-finite {key}')
        return value
    if dataset == 'MME':
        perception, reasoning = number('perception'), number('reasoning')
        return dict(score=perception + reasoning, perception=perception, reasoning=reasoning)
    if dataset == 'OCRBench':
        score = number('Final Score Norm') if 'Final Score Norm' in result else number('Final Score') / 10
    else:
        score = number('Overall') * 100
    if not 0 <= score <= 100:
        raise ValueError(f'{dataset}: percentage score outside [0,100]: {score}')
    return dict(score=score)


def export_result(path, model, dataset, result):
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    data.setdefault(model, {})[dataset] = normalize_result(dataset, result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def read_scores(path, benchmarks):
    data = json.loads(Path(path).read_text(encoding='utf-8'))['deepseek_vl2_small']
    missing = set(benchmarks) - data.keys()
    if missing:
        raise ValueError('Missing evaluation results: ' + ', '.join(sorted(missing)))
    scores = {name: float(data[name]['score']) for name in benchmarks}
    if 'MME' in benchmarks:
        scores.update({'MME_perception': float(data['MME']['perception']),
                       'MME_reasoning': float(data['MME']['reasoning'])})
    if all(name in scores for name in AVERAGE_BENCHMARKS):
        scores['Avg'] = statistics.mean(scores[name] for name in AVERAGE_BENCHMARKS)
    if not all(math.isfinite(v) for v in scores.values()):
        raise ValueError('Non-finite evaluation scores')
    return scores


def aggregate_runs(work, seeds, benchmarks):
    work = Path(work)
    runs = [{'seed': seed, 'metrics': read_scores(work / f'seed_{seed}/evaluation/metrics.json', benchmarks)}
            for seed in seeds]
    names = list(runs[0]['metrics'])
    summary = {}
    for name in names:
        values = [run['metrics'][name] for run in runs]
        summary[name] = dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else 0.,
                             values=values)
    result = dict(seeds=seeds, runs=runs, metrics=summary)
    (work / 'evaluation_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    with (work / 'evaluation_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['metric', *[f'seed_{s}' for s in seeds], 'mean', 'std'])
        for name in names:
            item = summary[name]
            writer.writerow([name, *item['values'], item['mean'], item['std']])
    return result

"""One Small-only entry: bank, estimate, solve, assemble, evaluate, or all."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .evaluation import DEFAULT_BENCHMARKS, aggregate_runs, read_scores

ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['bank', 'estimate', 'solve', 'assemble', 'evaluate', 'all'])
    p.add_argument('--model', help='Existing DeepSeek-VL2-Small checkpoint directory')
    p.add_argument('--data', help='Multimodal calibration/valuation JSONL')
    p.add_argument('--image-root', help='Local replacement for LMUData/images')
    p.add_argument('--work-dir', type=Path, default=ROOT / 'small_gptq_runs')
    p.add_argument('--bank', type=Path, help='Existing fixed GPTQ bank for later stages')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    p.add_argument('--nsamples', type=int, default=128)
    p.add_argument('--seqlen', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--permutations', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='Independent seeds for all (default: 0 1 2)')
    p.add_argument('--casp', type=float, default=.5)
    p.add_argument('--group-size', type=int, default=128)
    p.add_argument('--average-bits', type=float, default=2.6, help='Routed expert parameter-weighted average only')
    p.add_argument('--benchmarks', nargs='+', default=list(DEFAULT_BENCHMARKS))
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser(); args = p.parse_args(argv)
    if not 2 <= args.average_bits <= 4:
        p.error('--average-bits must be between 2 and 4 (routed experts)')
    if min(args.nsamples, args.seqlen, args.batch_size, args.permutations) <= 0 or args.seqlen < 2:
        p.error('Sample, sequence, batch and permutation counts must be positive; sequence >= 2')
    if args.group_size != 128:
        p.error('The aligned Small workflow uses group-size 128')
    if not 0 <= args.casp <= 1:
        p.error('--casp must be between 0 and 1')
    if args.stage in ['bank', 'all'] and (not args.model or not args.data):
        p.error('bank/all require --model and --data')
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        p.error('--seeds must contain distinct nonnegative integers')
    if args.stage == 'all' and args.bank:
        p.error('all builds a separate bank per seed; use --bank with individual stages')
    work = args.work_dir.resolve(); work.mkdir(parents=True, exist_ok=True)
    if args.stage == 'all':
        for seed in args.seeds:
            run_work = work / f'seed_{seed}'
            for stage in ['bank', 'estimate', 'solve', 'assemble', 'evaluate']:
                # Override the requested stage, seed, and output in isolated processes.
                child = [sys.executable, str(ROOT / 'run_small.py'), *argv,
                         '--work-dir', str(run_work), '--seed', str(seed)]
                child[2 + argv.index('all')] = stage
                subprocess.run(child, check=True)
        summary = aggregate_runs(work, args.seeds, args.benchmarks)
        print(json.dumps(summary['metrics'], indent=2), flush=True)
        return 0
    bank_path = (args.bank or work / 'bank').resolve()
    values_path, assignment_path = work / 'configuration_values.json', work / 'assignment.json'
    stages = ['bank', 'estimate', 'solve', 'assemble', 'evaluate'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        print(f'== {stage} ==', flush=True)
        if stage == 'evaluate':
            metrics_path = work / 'evaluation/metrics.json'
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text('{}', encoding='utf-8')
            cmd = [sys.executable, str(ROOT / 'newest/VLMEvalKit/run.py'),
                   '--model', 'deepseek_vl2_small', '--model_precision', 'fixed_gptq',
                   '--gptq_bank', str(bank_path), '--json_file', str(assignment_path),
                   '--model_device', args.device, '--work-dir', str(work / 'evaluation'),
                   '--metrics-file', str(metrics_path),
                   '--data', *args.benchmarks, '--mode', 'all', '--verbose']
            env = dict(os.environ, PYTHONPATH=str(ROOT / 'pipeline') + os.pathsep + os.environ.get('PYTHONPATH', ''))
            subprocess.run(cmd, check=True, cwd=ROOT / 'newest/VLMEvalKit', env=env)
            print(json.dumps(read_scores(metrics_path, args.benchmarks), indent=2), flush=True)
            continue
        import torch
        if stage != 'solve' and (not args.device.startswith('cuda') or not torch.cuda.is_available()):
            raise RuntimeError('A CUDA GPTQ/model environment is required; no CPU/RTN fallback')
        if stage != 'solve':
            torch.cuda.set_device(torch.device(args.device))
        if stage == 'bank':
            from .model import load_model, read_samples
            from .build import build_bank
            torch.manual_seed(args.seed)
            model, processor = load_model(args.model, args.device, args.dtype)
            samples = read_samples(processor, args.data, args.nsamples, args.seed, args.image_root)
            build_bank(model, processor, samples, bank_path, args.model, args.seqlen,
                       args.group_size, quantize_vision=True)
            del model, processor, samples
        elif stage == 'estimate':
            from .model import load_model, read_samples, embeddings, value
            from .bank import GPTQBank
            from .sampling import profile_and_sample
            from .estimator import ConfigOwenEstimator
            bank = GPTQBank(bank_path)
            torch.manual_seed(args.seed)
            model, processor = load_model(bank.manifest['model_path'], args.device, bank.manifest['dtype'])
            bank.bind(model)
            if args.data:
                samples = read_samples(processor, args.data, args.nsamples, args.seed, args.image_root)
                inputs = embeddings(model, samples, args.seqlen)
            else:
                inputs = torch.load(bank_path / 'calibration.pt', weights_only=True)
            prior, permutations = profile_and_sample(model, inputs, args.permutations, args.seed)
            batch_size = args.batch_size if len({x.shape[1] for x, _ in inputs}) == 1 else 1
            estimator = ConfigOwenEstimator(num_experts=len(prior), bits_high=4, bits_low=2,
                                            group_size=args.group_size, early_stop_threshold=args.casp,
                                            device=torch.device(args.device))
            values = estimator.estimate_configuration_values(
                model, inputs, permutations, prior, 64, 1, value,
                batch_size=batch_size)
            values_path.write_text(json.dumps(values, indent=2), encoding='utf-8')
            del model, processor
        elif stage == 'solve':
            from .bank import GPTQBank
            from .solve import solve
            bank = GPTQBank(bank_path)
            assignment, summary = solve(json.loads(values_path.read_text(encoding='utf-8')), bank, args.average_bits)
            assignment_path.write_text(json.dumps(assignment, indent=2), encoding='utf-8')
            (work / 'assignment_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            print(json.dumps(summary, indent=2), flush=True)
        elif stage == 'assemble':
            from .runtime import load_selected_model
            output = work / 'assembled'
            if output.exists():
                raise FileExistsError(f'Choose a new work directory or remove the old assembled output: {output}')
            model, processor, _ = load_selected_model(bank_path, assignment_path, args.device)
            model.save_pretrained(output, safe_serialization=True)
            processor.save_pretrained(output)
            (output / 'fixed_gptq.json').write_text(json.dumps(
                dict(bank=str(bank_path), assignment=str(assignment_path), representation='dequantized',
                     budget_scope='routed_experts'), indent=2), encoding='utf-8')
            del model, processor
        if torch.cuda.is_available():
            import gc
            gc.collect(); torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

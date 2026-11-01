"""Run the Small model on its six image benchmarks."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def parse_args():
    parser = argparse.ArgumentParser(description='DeepSeek-VL2-Small benchmark evaluation')
    parser.add_argument('--model', nargs='+', choices=['deepseek_vl2_small'], required=True)
    parser.add_argument('--data', nargs='+', required=True,
                        choices=['MME','AI2D_TEST','MMBench_DEV_EN_V11','MMStar','MMMU_DEV_VAL','OCRBench'])
    parser.add_argument('--model_precision', choices=['fixed_gptq','fp_baseline'], default='fixed_gptq')
    parser.add_argument('--gptq_bank')
    parser.add_argument('--json_file')
    parser.add_argument('--model_device', default='cuda:0')
    parser.add_argument('--model_path')
    parser.add_argument('--metrics-file')
    parser.add_argument('--work-dir', default='./outputs')
    parser.add_argument('--mode', choices=['all','infer','eval'], default='all')
    parser.add_argument('--judge', choices=['chatgpt-0125','gpt-4-0125','exact_matching'], default='chatgpt-0125')
    parser.add_argument('--api-nproc', type=int, default=4)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()

def main():
    import torch
    import torch.distributed as dist
    from vlmeval.config import supported_VLM
    from vlmeval.dataset import build_dataset
    from vlmeval.inference import infer_data_job
    from vlmeval.smp import get_rank_and_world_size, load_env
    from owen.evaluation.metrics import export_result
    args = parse_args()
    load_env()
    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        args.model_device = f'cuda:{local_rank}'
        dist.init_process_group(backend='nccl')
    try:
        for model_name in args.model:
            model = None
            work = Path(args.work_dir) / model_name
            work.mkdir(parents=True, exist_ok=True)
            for name in args.data:
                dataset = build_dataset(name)
                result_file = work / f'{model_name}_{name}.xlsx'
                if args.mode != 'eval':
                    model = model or supported_VLM[model_name](args=args)
                    model = infer_data_job(model, args, str(work), model_name, dataset,
                                           verbose=args.verbose, api_nproc=args.api_nproc)
                if rank == 0 and args.mode != 'infer':
                    if not result_file.is_file():
                        raise FileNotFoundError(result_file)
                    result = dataset.evaluate(str(result_file), model=args.judge,
                                              nproc=args.api_nproc, verbose=args.verbose, retry=3)
                    if args.metrics_file:
                        export_result(args.metrics_file, model_name, name, result)
                    print(name, result, flush=True)
                if world_size > 1:
                    dist.barrier()
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    main()

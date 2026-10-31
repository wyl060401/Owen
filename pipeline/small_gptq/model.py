"""DeepSeek-VL2-Small loading and one consistent assistant-token loss."""
import json
from pathlib import Path
import random
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]


def select_calibration_rows(rows, nsamples=128, seed=0):
    """Require all five paper sources; sample without replacement."""
    families = ('DocVQA', 'InfoVQA', 'ScienceQA', 'TextVQA', 'ChartQA')
    pools = {name: [] for name in families}
    for row in rows:
        source = row.get('source', '')
        family = next((n for n in families if source == n or source.startswith(n + '_')), None)
        if family is None:
            raise ValueError(f'Calibration source {source!r} is outside the five paper datasets; rebuild calibration JSONL')
        pools[family].append(row)
    missing = [n for n in families if not pools[n]]
    if missing:
        raise ValueError('Missing paper calibration sources: ' + ', '.join(missing))
    if nsamples < 5 or len(rows) < nsamples:
        raise ValueError(f'Need >=5 and <= {len(rows)} samples, requested {nsamples}')
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    selected = []
    while len(selected) < nsamples:
        for pool in pools.values():
            if pool and len(selected) < nsamples:
                selected.append(pool.pop())
    rng.shuffle(selected)
    return selected


def load_model(model_path, device='cuda:0', dtype='bfloat16', validate_small=True):
    sys.path.insert(0, str(ROOT / 'newest/VLMEvalKit/vlmeval/vlm'))
    from DeepSeek_VL2.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM
    from DeepSeek_VL2.processing_deepseek_vl_v2 import DeepseekVLV2Processor
    config = json.loads((Path(model_path) / 'config.json').read_text(encoding='utf-8'))
    language = config.get('language_config', {})
    if validate_small and (language.get('hidden_size'), language.get('n_routed_experts'),
                           language.get('num_hidden_layers')) != (2048, 64, 27):
        raise ValueError('This workflow supports DeepSeek-VL2-Small (2048, 64 experts, 27 layers) only')
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        str(model_path), torch_dtype=getattr(torch, dtype), device_map={'': device},
        local_files_only=True, trust_remote_code=True).eval()
    processor = DeepseekVLV2Processor.from_pretrained(str(model_path), local_files_only=True)
    return model, processor


def read_samples(processor, data, nsamples, seed=0, image_root=None):
    from PIL import Image
    data = Path(data).resolve()
    rows = [json.loads(line) for line in data.read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = select_calibration_rows(rows, nsamples, seed)
    result = []
    for row in rows:
        path = Path(str(row['image_path']))
        if not path.is_file():
            norm = str(row['image_path']).replace('\\', '/')
            relative = norm.split('/LMUData/images/', 1)[-1]
            path = Path(image_root) / relative if image_root else data.parent / path
        if not path.is_file():
            raise FileNotFoundError(f'Calibration image does not exist: {path}; use --image-root')
        conversations = [{'role': {'user': '<|User|>', 'assistant': '<|Assistant|>'}.get(t['role'], t['role']),
                          'content': t['content']} for t in row['conversations']]
        with Image.open(path) as image:
            prepared = processor(conversations=conversations, images=[image.convert('RGB')],
                                 force_batchify=True, system_prompt='')
        result.append({key: getattr(prepared, key).cpu() for key in
                       ['input_ids', 'images', 'images_seq_mask', 'images_spatial_crop', 'attention_mask', 'labels']})
    return result


def to_model_inputs(model, sample):
    parameter = next(model.parameters())
    return {k: v.to(device=parameter.device, dtype=parameter.dtype if v.is_floating_point() else v.dtype)
            for k, v in sample.items()}


def embeddings(model, samples, seqlen=2048):
    device = next(model.parameters()).device
    result = []
    with torch.no_grad():
        for sample in samples:
            batch = to_model_inputs(model, sample)
            x = model.prepare_inputs_embeds(**batch)[:, -seqlen:]
            labels = batch['labels'][:, -seqlen:]
            if not (labels[:, 1:] != -100).any():
                raise ValueError('Sample has no assistant loss tokens after truncation')
            result.append((x.detach().cpu(), labels.detach().cpu()))
    return result


def value(model, inputs):
    """Same shifted-token weighting as the cached configuration evaluator."""
    device = next(model.parameters()).device
    total, tokens = 0., 0
    with torch.no_grad():
        for embeds, labels in inputs:
            labels = labels.to(device)
            outputs = model.language(inputs_embeds=embeds.to(device), use_cache=False)
            logits = outputs.logits[..., :-1, :].float()
            targets = labels[..., 1:]
            n = int((targets != -100).sum())
            if n:
                total += torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction='sum').item()
                tokens += n
    if not tokens:
        raise ValueError('No valid shifted labels')
    import math
    return math.exp(-min(total / tokens, 100.))

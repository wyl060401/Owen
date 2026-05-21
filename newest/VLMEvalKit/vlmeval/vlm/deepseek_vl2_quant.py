
import os
import sys
import argparse
from .DeepSeek_VL2.processing_deepseek_vl_v2 import DeepseekVLV2Processor
from .DeepSeek_VL2.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM
from .DeepSeek_VL2.io_utils import load_pil_images
import torch
from auto_round import AutoRoundMLLM
from auto_round.utils import get_fp_layer_names, clear_memory, logger
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoProcessor
from auto_round.utils import check_awq_gemm_compatibility

def quantized_model(model_name='deepseek-ai/deepseek-vl2-tiny', eval=False, bits=4, eval_bs=None, device='0', asym=False, dataset=None,
         lr=None, minmax_lr=None, seed=42, adam=False, gradient_accumulate_steps=1, nblocks=1, low_gpu_mem_usage=False,
         format='auto_round', data_type='int', scale_dtype='fp16', output_dir='./tmp_autoround', disable_amp=False, disable_minmax_tuning=False,
         enable_norm_bias_tuning=False, disable_trust_remote_code=False, disable_quanted_input=False, quant_lm_head=False, low_cpu_mem_mode=0, 
         low_cpu_mem_tmp_dir=None, model_dtype=None, act_bits=32, fp_layers='', not_use_best_mse=False, disable_torch_compile=False, 
         disable_deterministic_algorithms=False, quant_nontext_module=False, extra_data_dir=None, template=None, truncation=False, to_quant_block_names=None,
         device_map = None, group_size=128, batch_size=8, iters=200, seqlen=None, nsamples=128):
    
    cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'
    os.environ['HF_HOME'] = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'
    os.environ['HF_DATASETS_CACHE'] = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'
    os.environ['TRANSFORMERS_CACHE'] = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

    if format is None:
        format = "auto_round"
    supported_formats = ["auto_round", "auto_round:auto_gptq", "auto_round:auto_awq", "auto_awq", "fake"]
    if not quant_nontext_module:
        supported_formats.extend(["auto_gptq", "auto_gptq:marlin"])

    formats = format.replace(' ', '').split(",")
    for format in formats:
        if format not in supported_formats:
            raise ValueError(f"{format} is not supported, we only support {supported_formats}")

    # torch_dtype = "auto"

    processor, image_processor = None, None

    model = DeepseekVLV2ForCausalLM.from_pretrained(model_name, 
                                                    trust_remote_code=True, 
                                                    cache_dir = cache_dir, 
                                                    device_map=device_map
                                                    )
    for param in model.parameters():
        param.requires_grad = True

    model.train()
    
    processor = DeepseekVLV2Processor.from_pretrained(model_name, trust_remote_code=True, cache_dir = cache_dir)
    tokenizer = processor.tokenizer

    model = model.eval()

    if model_dtype != None:
        try:
            if model_dtype == "float16" or model_dtype == "fp16":
                model = model.to(torch.float16)
            elif model_dtype == "bfloat16" or model_dtype == "bfp16" or model_dtype == "bf16":
                model = model.to(torch.bfloat16)
            elif model_dtype == "float32" or model_dtype == "fp32":
                model = model.to(torch.float32)
        except:
            logger.error("please use more device to fit the device or just use one device")
            exit()

    round = AutoRoundMLLM

    layer_config = {}
    not_quantize_layer_names = get_fp_layer_names(model, fp_layers)
    for name in not_quantize_layer_names:
        layer_config[name] = {"bits": 16}
    if len(not_quantize_layer_names) > 0:
        logger.info(f"{not_quantize_layer_names} will not be quantized.")
        for format in formats:
            if "auto_round" not in format and "fake" not in format and "awq" not in format:
                logger.warning(f"mixed precision exporting does not support {format} currently")

    layer_config = {}
    if fp_layers != "":
        fp_layers = fp_layers.replace(" ", "").split(",")
        for n, m in model.named_modules():
            if not isinstance(m, (torch.nn.Linear, transformers.modeling_utils.Conv1D)):
                continue
            for fp_layer in fp_layers:
                if fp_layer in n:
                    layer_config[n] = {"bits": 16}
                    logger.info(f"{n} will not be quantized.")
        if len(layer_config) > 0:
            for format in formats:
                if "auto_round" not in format and "fake" not in format:
                    logger.warning(f"mixed precision exporting does not support {format} currently")

    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, transformers.modeling_utils.Conv1D):
            if m.weight.shape[0] % 32 != 0 or m.weight.shape[1] % 32 != 0:
                layer_config[n] = {"bits": 32}
                logger.info(
                    f"{n} will not be quantized due to its shape not being divisible by 32,"
                    " resulting in an exporting issue to autogptq")
                
    lm_head_layer_name = "lm_head"
    for n, _ in model.named_modules():
        lm_head_layer_name = n
    if quant_lm_head:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=not disable_trust_remote_code)
        if config.tie_word_embeddings and hasattr(model, "_tied_weights_keys"):
            tied_keys = model._tied_weights_keys
            for item in tied_keys:
                if lm_head_layer_name in item:
                    quant_lm_head = False
                    print(
                        f"warning, disable quant_lm_head as quantizing lm_head with tied weights has not been "
                        f"supported currently")
                    break

    if quant_lm_head:
        layer_config[lm_head_layer_name] = {"bits": bits}
        for format in formats:
            if "auto_round" not in format and "fake" not in format:
                auto_round_formats = [s for s in supported_formats if s.startswith("auto_round")]
                raise ValueError(
                    f"{format} is not supported for lm-head quantization, please change to {auto_round_formats}")

    if "--truncation" not in sys.argv:
        truncation = None

    if "auto_awq" in format:
        
        awq_supported, info = check_awq_gemm_compatibility(
            model, bits, group_size, not asym, layer_config)
        if not awq_supported:
            logger.warning(f"The AutoAWQ format may not be supported due to {info}")

    enable_torch_compile = False if "--disable_torch_compile" in sys.argv else None

    # import random
    # for i in range(1, 12):
    #     for j in range(0, 64):
    #         rand_choice = random.choice([2, 4, 8])

    #         layer_config[f"language.model.layers.{i}.mlp.experts.{j}.up_proj"] = {'bits': rand_choice, 'group_size': 64}
    #         layer_config[f"language.model.layers.{i}.mlp.experts.{j}.down_proj"] = {'bits': rand_choice, 'group_size': 64}
    #         layer_config[f"language.model.layers.{i}.mlp.experts.{j}.gate_proj"] = {'bits': rand_choice, 'group_size': 64}

    autoround = round(
        model,
        tokenizer,
        processor=processor,
        image_processor=image_processor,
        dataset=dataset,
        extra_data_dir=extra_data_dir,
        bits=bits,
        group_size=group_size,
        sym=not asym,
        batch_size=batch_size,
        seqlen=seqlen,
        nblocks=nblocks,
        iters=iters,
        lr=lr,
        minmax_lr=minmax_lr,
        amp=not disable_amp,
        enable_quanted_input=not disable_quanted_input,
        truncation=truncation,
        nsamples=nsamples,
        low_gpu_mem_usage=low_gpu_mem_usage,
        seed=seed,
        gradient_accumulate_steps=gradient_accumulate_steps,
        scale_dtype=scale_dtype,
        layer_config=layer_config,
        template=template,
        enable_minmax_tuning=not disable_minmax_tuning,
        act_bits=act_bits,
        quant_nontext_module=quant_nontext_module,
        not_use_best_mse=not_use_best_mse,
        to_quant_block_names=to_quant_block_names,
        enable_torch_compile=enable_torch_compile,
        device_map=device_map)
    model, layer_config = autoround.quantize()

    for keydict in layer_config:
        print(keydict, layer_config[keydict])

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    clear_memory()

    
    
    return model, processor, tokenizer

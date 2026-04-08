import os
import sys
import warnings

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from .base import BaseModel
from ..smp import *


def split_model(model_name):
    normalized = str(model_name).lower().replace("\\", "/")
    if (
        model_name == "deepseek-ai/deepseek-vl2-tiny"
        or "deepseek-vl2-tiny" in normalized
        or "deepseek_vl2_tiny" in normalized
    ):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return f"cuda:{local_rank}"

    device_map = {}
    model_splits = {
        "deepseek-ai/deepseek-vl2-small": [13, 14],
        "deepseek-ai/deepseek-vl2": [10, 10, 10],
    }
    num_layers_per_gpu = model_splits[model_name]
    num_layers = sum(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            device_map[f"language.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision"] = 0
    device_map["projector"] = 0
    device_map["image_newline"] = 0
    device_map["view_seperator"] = 0
    device_map["language.model.embed_tokens"] = 0
    device_map["language.model.norm"] = 0
    device_map["language.lm_head"] = 0
    device_map[f"language.model.layers.{num_layers - 1}"] = 0
    return device_map


class DeepSeekVL2(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path="deepseek-ai/deepseek-vl2-tiny", args=None, **kwargs):
        assert model_path is not None

        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_this_dir))))

        override_model_path = os.environ.get("DEEPSEEK_VL2_TINY_MODEL_PATH")
        if override_model_path:
            model_path = override_model_path
        elif model_path == "deepseek-ai/deepseek-vl2-tiny":
            model_path = os.path.join(_project_root, "models/deepseek-vl2-tiny")

        self.model_path = model_path
        is_tiny_path = "deepseek-vl2-tiny" in model_path or "deepseek_vl2_tiny" in model_path
        self.device_map = split_model("deepseek-ai/deepseek-vl2-tiny" if is_tiny_path else model_path)
        self.main_device = self.device_map if isinstance(self.device_map, str) else "cuda:0"

        model_precision = getattr(args, "model_precision", "") if args is not None else ""
        quantized_model_path = (
            os.path.isdir(model_path)
            and os.path.exists(os.path.join(model_path, "model.safetensors"))
            and os.path.exists(os.path.join(model_path, "quant_config_original.json"))
        )

        if model_precision == "uniform_quant":
            torch.set_grad_enabled(True)
            torch.enable_grad()

            from .deepseek_vl2_quant import quantized_model

            model, self.vl_chat_processor, self.tokenizer = quantized_model(
                bits=args.bits,
                model_name=model_path,
                format=args.quant_format,
                device_map=self.device_map,
            )

            torch.set_grad_enabled(False)
            self.model = model.to(torch.bfloat16).eval()

        elif model_precision == "mixed_precision_quant":
            torch.set_grad_enabled(True)
            torch.enable_grad()

            from .deepseek_vl2_quant import quantized_model

            json_file = getattr(args, "json_file", None)
            model, self.vl_chat_processor, self.tokenizer = quantized_model(
                bits=args.bits,
                model_name=model_path,
                format=args.quant_format,
                device_map=self.device_map,
                json_file=json_file,
                calib_source=getattr(args, "calib_source", "pile10k"),
            )

            torch.set_grad_enabled(False)
            self.model = model.to(torch.bfloat16).eval()

        elif model_precision == "activation_frequency_profiling":
            from .DeepSeek_VL2_quant.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM
            from .DeepSeek_VL2_quant.processing_deepseek_vl_v2 import DeepseekVLV2Processor

            self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
            self.tokenizer = self.vl_chat_processor.tokenizer

            self.model = DeepseekVLV2ForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                cache_dir=os.path.join(_project_root, "models"),
            )

        elif model_precision == "sensitivity_profiling":
            from .DeepSeek_VL2_quant.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM
            from .DeepSeek_VL2_quant.processing_deepseek_vl_v2 import DeepseekVLV2Processor

            self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
            self.tokenizer = self.vl_chat_processor.tokenizer

            self.fp_model = DeepseekVLV2ForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                cache_dir=os.path.join(_project_root, "models"),
            )

        elif model_precision == "fp_baseline":
            from .deepseek_model.processing_deepseek_vl_v2 import DeepseekVLV2Processor
            from .deepseek_model.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM

            self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
            self.tokenizer = self.vl_chat_processor.tokenizer

            model = DeepseekVLV2ForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                cache_dir=os.path.join(_project_root, "models"),
            )
            self.model = model.eval()

        elif quantized_model_path:
            from .DeepSeek_VL2.processing_deepseek_vl_v2 import DeepseekVLV2Processor
            from .mixedbit_gptq_loader import load_mixedbit_quantized_deepseek_vl2

            self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
            self.tokenizer = self.vl_chat_processor.tokenizer
            self.model = load_mixedbit_quantized_deepseek_vl2(
                model_path=model_path,
                device=self.main_device,
                torch_dtype=torch.bfloat16,
            )

        else:
            from .deepseek_model.processing_deepseek_vl_v2 import DeepseekVLV2Processor
            from .deepseek_model.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM

            self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
            self.tokenizer = self.vl_chat_processor.tokenizer
            self.model = DeepseekVLV2ForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
            ).eval()

        torch.cuda.empty_cache()
        default_kwargs = dict(max_new_tokens=512, do_sample=False, use_cache=True)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def prepare_inputs(self, message, dataset=None):
        if dataset == "MMMU_DEV_VAL":

            def prepare_itlist(msgs):
                content, images = "", []
                image_idx = 1
                for s in msgs:
                    if s["type"] == "image":
                        images.append(s["value"])
                        content += f"<image {image_idx}>"
                        image_idx += 1
                    elif s["type"] == "text":
                        content += s["value"]
                content = "<image>" * (image_idx - 1) + "\n" + content
                return content, images

            conversation = []
            if "role" not in message[0]:
                content, images = prepare_itlist(message)
                content = content.replace(
                    "Please select the correct answer from the options above.",
                    "Answer with the option's letter from the given choices directly. Answer the question using a single word or phrase.\n",
                )
                content = content.replace("Question:", "")
                content = content.replace("Options:\n", "")
                conversation.append(dict(role="<|User|>", content=content, images=images))
            else:
                role_map = {"user": "<|User|>", "assistant": "<|Assistant|>"}
                for msgs in message:
                    role = role_map[msgs["role"]]
                    content, images = prepare_itlist(msgs["content"])
                    content = content.replace(
                        "Please select the correct answer from the options above.",
                        "Answer with the option's letter from the given choices directly. Answer the question using a single word or phrase.\n",
                    )
                    content = content.replace("Question:", "")
                    content = content.replace("Options:\n", "")
                    conversation.append(dict(role=role, content=content, images=images))
            conversation.append(dict(role="<|Assistant|>", content=""))

        else:

            def prepare_itlist(msgs):
                content, images = "", []
                for s in msgs:
                    if s["type"] == "image":
                        images.append(s["value"])
                        content += "<image>\n"
                    elif s["type"] == "text":
                        content += s["value"]
                return content, images

            conversation = []
            if "role" not in message[0]:
                content, images = prepare_itlist(message)
                conversation.append(dict(role="<|User|>", content=content, images=images))
            else:
                role_map = {"user": "<|User|>", "assistant": "<|Assistant|>"}
                for msgs in message:
                    role = role_map[msgs["role"]]
                    content, images = prepare_itlist(msgs["content"])
                    conversation.append(dict(role=role, content=content, images=images))
            conversation.append(dict(role="<|Assistant|>", content=""))

        return conversation

    def generate_inner(self, message, dataset=None):
        conversation = self.prepare_inputs(message, dataset)
        from .deepseek_model.io_utils import load_pil_images

        pil_images = load_pil_images(conversation)

        if dataset == "MMMU_DEV_VAL" and len(pil_images):
            h, w = pil_images[0].size
            pil_images[0] = pil_images[0].resize((2 * h, 2 * w), Image.BILINEAR)

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt="",
        )

        gen_kwargs = dict(self.kwargs)
        short_answer_limits = {
            "MME": 32,
            "BLINK": 32,
            "RealWorldQA": 32,
            "DocVQA_VAL": 32,
            "InfoVQA_VAL": 32,
            "ScienceQA_VAL": 16,
            "MMBench_DEV_EN_V11": 16,
            "POPE": 8,
            "HallusionBench": 8,
        }
        if dataset in short_answer_limits:
            gen_kwargs["max_new_tokens"] = min(
                gen_kwargs.get("max_new_tokens", 512),
                short_answer_limits[dataset],
            )

        prepare_inputs = prepare_inputs.to(self.main_device)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        inputs_embeds, past_key_values = self.model.incremental_prefilling(
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            chunk_size=512,
        )

        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=past_key_values,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )

        answer = self.tokenizer.decode(
            outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
            skip_special_tokens=True,
        )
        answer = answer.rstrip(".")

        return answer

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message, dataset=dataset)

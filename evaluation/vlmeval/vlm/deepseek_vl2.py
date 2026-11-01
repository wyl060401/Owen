"""DeepSeek-VL2-Small adapter for the selected fixed GPTQ candidates."""
from pathlib import Path
import sys
import warnings
import torch
from PIL import Image
from .base import BaseModel
from ..smp import *

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

class DeepSeekVL2(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='deepseek-ai/deepseek-vl2-small', args=None, **kwargs):
        precision = getattr(args, 'model_precision', 'fp_baseline') or 'fp_baseline'
        device = getattr(args, 'model_device', 'cuda:0')
        if precision == 'fixed_gptq':
            from owen.models.runtime import load_selected_model
            bank = getattr(args, 'gptq_bank', None)
            assignment = getattr(args, 'json_file', None)
            if not bank or not assignment or assignment == 'None':
                raise ValueError('fixed_gptq requires --gptq_bank and --json_file')
            self.model, self.vl_chat_processor, self.tokenizer = load_selected_model(bank, assignment, device)
        elif precision == 'fp_baseline':
            from owen.models.model import load_model
            self.model, self.vl_chat_processor = load_model(getattr(args, 'model_path', None) or model_path, device)
            self.tokenizer = self.vl_chat_processor.tokenizer
        else:
            raise ValueError('Small supports fixed_gptq or fp_baseline; no online re-quantization')
        self.main_device = next(self.model.parameters()).device
        self.kwargs = dict(max_new_tokens=512, do_sample=False, use_cache=True)
        self.kwargs.update(kwargs)

    def prepare_inputs(self, message, dataset=None):
        if dataset == 'MMMU_DEV_VAL':

            def prepare_itlist(msgs):
                content, images = ('', [])
                image_idx = 1
                for s in msgs:
                    if s['type'] == 'image':
                        images.append(s['value'])
                        content += f'<image {image_idx}>'
                        image_idx += 1
                    elif s['type'] == 'text':
                        content += s['value']
                content = '<image>' * (image_idx - 1) + '\n' + content
                return (content, images)
            conversation = []
            if 'role' not in message[0]:
                content, images = prepare_itlist(message)
                content = content.replace('Please select the correct answer from the options above.', "Answer with the option's letter from the given choices directly. Answer the question using a single word or phrase.\n")
                content = content.replace('Question:', '')
                content = content.replace('Options:\n', '')
                conversation.append(dict(role='<|User|>', content=content, images=images))
            else:
                role_map = {'user': '<|User|>', 'assistant': '<|Assistant|>'}
                for msgs in message:
                    role = role_map[msgs['role']]
                    content, images = prepare_itlist(msgs['content'])
                    content = content.replace('Please select the correct answer from the options above.', "Answer with the option's letter from the given choices directly. Answer the question using a single word or phrase.\n")
                    content = content.replace('Question:', '')
                    content = content.replace('Options:\n', '')
                    conversation.append(dict(role=role, content=content, images=images))
            conversation.append(dict(role='<|Assistant|>', content=''))
        else:

            def prepare_itlist(msgs):
                content, images = ('', [])
                for s in msgs:
                    if s['type'] == 'image':
                        images.append(s['value'])
                        content += '<image>\n'
                    elif s['type'] == 'text':
                        content += s['value']
                return (content, images)
            conversation = []
            if 'role' not in message[0]:
                content, images = prepare_itlist(message)
                conversation.append(dict(role='<|User|>', content=content, images=images))
            else:
                role_map = {'user': '<|User|>', 'assistant': '<|Assistant|>'}
                for msgs in message:
                    role = role_map[msgs['role']]
                    content, images = prepare_itlist(msgs['content'])
                    conversation.append(dict(role=role, content=content, images=images))
            conversation.append(dict(role='<|Assistant|>', content=''))
        return conversation

    def generate_inner(self, message, dataset=None):
        conversation = self.prepare_inputs(message, dataset)
        from owen.models.deepseek_vl2.io_utils import load_pil_images
        pil_images = load_pil_images(conversation)
        if dataset == 'MMMU_DEV_VAL':
            if len(pil_images):
                h, w = pil_images[0].size
                pil_images[0] = pil_images[0].resize((2 * h, 2 * w), Image.BILINEAR)
        prepare_inputs = self.vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True, system_prompt='')
        prepare_inputs = prepare_inputs.to(self.main_device, dtype=next(self.model.parameters()).dtype)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        inputs_embeds, past_key_values = self.model.incremental_prefilling(input_ids=prepare_inputs.input_ids, images=prepare_inputs.images, images_seq_mask=prepare_inputs.images_seq_mask, images_spatial_crop=prepare_inputs.images_spatial_crop, attention_mask=prepare_inputs.attention_mask, chunk_size=512)
        outputs = self.model.generate(inputs_embeds=inputs_embeds, input_ids=prepare_inputs.input_ids, images=prepare_inputs.images, images_seq_mask=prepare_inputs.images_seq_mask, images_spatial_crop=prepare_inputs.images_spatial_crop, attention_mask=prepare_inputs.attention_mask, past_key_values=past_key_values, pad_token_id=self.tokenizer.eos_token_id, bos_token_id=self.tokenizer.bos_token_id, eos_token_id=self.tokenizer.eos_token_id, **self.kwargs)
        answer = self.tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=True)
        answer = answer.rstrip('.')
        return answer

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message, dataset=dataset)

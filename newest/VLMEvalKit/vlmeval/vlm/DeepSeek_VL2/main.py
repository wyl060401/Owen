

if __name__ == "__main__":
    import torch
    # from transformers import AutoModelForCausalLM
    
    from processing_deepseek_vl_v2 import DeepseekVLV2Processor
    from io_utils import load_pil_images
    from modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM 

    model_path = "deepseek-ai/deepseek-vl2-tiny"
    
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer         = vl_chat_processor.tokenizer

    cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

    vl_gpt = DeepseekVLV2ForCausalLM.from_pretrained(model_path, trust_remote_code=True, cache_dir = cache_dir)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\n<|ref|>The giraffe at the back.<|/ref|>.",
            "images": ["visual_grounding_1.jpeg"],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
    print(f"{prepare_inputs['sft_format'][0]}", answer)
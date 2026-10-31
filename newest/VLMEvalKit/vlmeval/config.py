from functools import partial
from vlmeval.vlm import DeepSeekVL2
supported_VLM = {'deepseek_vl2_small': partial(DeepSeekVL2, model_path='deepseek-ai/deepseek-vl2-small')}

from vlmeval.vlm import *
# from vlmeval.api import *
from functools import partial

PandaGPT_ROOT = None
MiniGPT4_ROOT = None
TransCore_ROOT = None
Yi_ROOT = None
OmniLMM_ROOT = None
Mini_Gemini_ROOT = None
VXVERSE_ROOT = None
VideoChat2_ROOT = None
VideoChatGPT_ROOT = None
PLLaVA_ROOT = None
RBDash_ROOT = None
VITA_ROOT = None
LLAVA_V1_7B_MODEL_PTH = 'Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. '

deepseekvl2_series = {
    'deepseek_vl2_tiny': partial(DeepSeekVL2, model_path='deepseek-ai/deepseek-vl2-tiny'),
    'deepseek_vl2_small': partial(DeepSeekVL2, model_path='deepseek-ai/deepseek-vl2-small'),
    'deepseek_vl2': partial(DeepSeekVL2, model_path='deepseek-ai/deepseek-vl2'),
}

molmo_series={
    'molmoE-1B-0924': partial(molmo, model_path='allenai/MolmoE-1B-0924')
    }

supported_VLM = {}

model_groups = [deepseekvl2_series, molmo_series]

for grp in model_groups:
    supported_VLM.update(grp)

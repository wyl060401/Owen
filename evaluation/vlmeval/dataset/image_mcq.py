import warnings
from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
import pandas as pd

class ImageMCQDataset(ImageBaseDataset):
    TYPE = 'MCQ'
    DATASET_URL = {'MMBench_DEV_EN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN_V11.tsv', 'AI2D_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv', 'MMStar': 'https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv'}
    DATASET_MD5 = {'MMBench_DEV_EN_V11': '30c05be8f2f347a50be25aa067248184', 'AI2D_TEST': '0f593e0d1c7df9a3d69bf1f947e71975', 'MMStar': 'e1ecd2140806c1b1bbf54b43372efb9e'}

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and (not pd.isna(line[cand]))}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if 'hint' in line and (not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import report_acc, report_acc_MMT, mcq_circular_eval, mcq_vanilla_eval
        dataset_map = {'MMBench_TEST_EN': 'MMBench', 'MMBench_TEST_EN_V11': 'MMBench_V11', 'MMBench_TEST_CN': 'MMBench_CN', 'MMBench_TEST_CN_V11': 'MMBench_CN_V11'}
        dataset = self.dataset_name
        if dataset in dataset_map:
            dataset = dataset_map[dataset]
        nproc = judge_kwargs.pop('nproc', 4)
        circular = False
        if listinstr(['mmbench', 'ccbench'], dataset.lower()):
            data = load(eval_file)
            data['index'] = [int(x) for x in data['index']]
            dump(data, eval_file)
            circular = True
        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model
        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None
        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')
        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)
        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
        if circular:
            data = mcq_circular_eval(model, data, meta, nproc, result_file, self.dataset_name)
        else:
            data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)
        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        if 'MMT' in dataset:
            acc = report_acc_MMT(data)
        else:
            acc = report_acc(data)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)
        if dataset == 'AesBench_VAL':
            warnings.warn('Note that AesBench VAL is just a toy version of AesBench TEST. For full results,                            please evaluate on AesBench TEST. The AesBench TEST dataset is more than 20 times                            larger than the VAL dataset and the leaderboard results are based on AesBench TEST.')
        if dataset == 'VisOnlyQA-VLMEvalKit':
            warnings.warn('Note that the results on VisOnlyQA-VLMEvalKit are different from the results on                            the original VisOnlyQA. VisOnlyQA-VLMEvalKit does not include the                            chemistry__shape_multi split and uses a different evaluation prompt. Please                            explicitly specify the version of the dataset when you report results.')
        return acc

class MMMUDataset(ImageMCQDataset):
    DATASET_URL = {'MMMU_DEV_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv'}
    DATASET_MD5 = {'MMMU_DEV_VAL': '585e8ad75e73f75dcad265dfd0417d64'}

    @staticmethod
    def split_MMMU(msgs):
        text, images = (None, [])
        for s in msgs:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                assert text is None
                text = s['value']
        text_segs = text.split('<image ')
        if len(text_segs) == 1:
            return msgs
        segs = [dict(type='text', value=text_segs[0])]
        for i, seg in enumerate(text_segs):
            if i == 0:
                continue
            assert istype(seg[0], int) and seg[1] == '>'
            image_idx = int(seg[0]) - 1
            segs.append(dict(type='image', value=images[image_idx]))
            segs.append(dict(type='text', value=seg[2:]))
        return segs

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        msgs = self.split_MMMU(msgs)
        return msgs

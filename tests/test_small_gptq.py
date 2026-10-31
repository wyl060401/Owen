"""CPU arithmetic, selection, and integration tests for the Small workflow."""
import ast
from contextlib import redirect_stdout
import copy
import io
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline'))
from small_gptq import backend, reference_quantizer
from small_gptq.build import candidate
from small_gptq.bank import GPTQBank
from small_gptq.estimator import ConfigOwenEstimator, enumerate_menus, menu_to_key
from small_gptq.model import value
from small_gptq.sampling import permutation_weights, profile_and_sample
from small_gptq.solve import solve


class Attention(nn.Module):
    def forward(self, x=None, hidden_states=None, **kwargs):
        x = x if x is not None else hidden_states
        return x * .1, None, None


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(3, 4, bias=False)
        self.up_proj = nn.Linear(3, 4, bias=False)
        self.down_proj = nn.Linear(4, 3, bias=False)
    def forward(self, x):
        return self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x))


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 3))
    def forward(self, x):
        weights = torch.softmax(x @ self.weight.t(), dim=-1)
        ids = torch.arange(2, device=x.device).expand(*x.shape[:-1], 2)
        return ids, weights, None


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_routed_experts=2)
        self.experts = nn.ModuleList([Expert(), Expert()])
        self.gate = Gate()
    def forward(self, x):
        _, weights, _ = self.gate(x)
        return sum(expert(x) * weights[..., i:i+1] for i, expert in enumerate(self.experts))


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.Identity(); self.post_attention_layernorm = nn.Identity()
        self.self_attn = Attention(); self.mlp = MoE()
    def forward(self, x, **kwargs):
        x = x + self.self_attn(x)[0]
        return (x + self.mlp(x),)


class Language(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module(); self.model.layers = nn.ModuleList([Layer(), Layer()])
        self.model.norm = nn.Identity(); self.model._use_flash_attention_2 = False
        self.lm_head = nn.Linear(3, 7, bias=False)
    def forward(self, inputs_embeds, **kwargs):
        h = inputs_embeds
        for layer in self.model.layers:
            h = layer(h)[0]
        return SimpleNamespace(logits=self.lm_head(h))


class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.language = Language()
    def prepare_inputs_embeds(self, inputs_embeds, **kwargs):
        return inputs_embeds


class SmallGPTQTests(unittest.TestCase):
    def test_small_evaluation_driver_runs_all_six_benchmarks_and_exports_metrics(self):
        import importlib.util
        sys.path.insert(0,str(ROOT/'newest/VLMEvalKit'))
        from small_gptq.evaluation import DEFAULT_BENCHMARKS, read_scores
        spec=importlib.util.spec_from_file_location('small_eval_run',ROOT/'newest/VLMEvalKit/run.py')
        driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);metrics=directory/'metrics.json'
            def dataset(name):
                score={'perception':1200,'reasoning':300} if name=='MME' else (
                    {'Final Score':500} if name=='OCRBench' else {'Overall':.5})
                return SimpleNamespace(dataset_name=name,evaluate=lambda *args,**kwargs:score)
            def infer(model,args,work_dir,model_name,dataset,**kwargs):
                (Path(work_dir)/f'{model_name}_{dataset.dataset_name}.xlsx').touch()
                return model
            constructor=SimpleNamespace()
            with patch('vlmeval.config.supported_VLM',{'deepseek_vl2_small':lambda **kw:constructor}), \
                 patch('vlmeval.dataset.build_dataset',side_effect=dataset) as build, \
                 patch('vlmeval.inference.infer_data_job',side_effect=infer) as run, \
                 patch('vlmeval.smp.get_rank_and_world_size',return_value=(0,1)), \
                 patch.object(sys,'argv',['run.py','--model','deepseek_vl2_small','--data',*DEFAULT_BENCHMARKS,
                     '--work-dir',str(directory),'--metrics-file',str(metrics)]), redirect_stdout(io.StringIO()):
                driver.main()
            self.assertEqual(build.call_count,6)
            self.assertEqual(run.call_count,6)
            self.assertEqual(read_scores(metrics,DEFAULT_BENCHMARKS)['Avg'],50)

    def test_small_evaluation_registry_and_real_scoring_survive_cleanup(self):
        sys.path.insert(0,str(ROOT/'newest/VLMEvalKit'))
        import pandas as pd
        from vlmeval.dataset import SUPPORTED_DATASETS, build_dataset, OCRBench, ImageMCQDataset, ImageYORNDataset
        from vlmeval.config import supported_VLM
        self.assertEqual(set(SUPPORTED_DATASETS),{'MME','AI2D_TEST','MMBench_DEV_EN_V11','MMStar','MMMU_DEV_VAL','OCRBench'})
        self.assertEqual(set(supported_VLM),{'deepseek_vl2_small'})
        with self.assertRaises(ValueError):
            build_dataset('POPE')
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            ocr=directory/'ocr.xlsx'
            pd.DataFrame([dict(prediction='Hello world',answer="['hello']",category='Regular Text Recognition')]).to_excel(ocr,index=False)
            score=OCRBench.evaluate(str(ocr))
            self.assertEqual(score['Final Score'],1)
            self.assertEqual(score['Final Score Norm'],.1)
            mcq=ImageMCQDataset.__new__(ImageMCQDataset)
            mcq.dataset_name='MMStar'
            mcq.data=pd.DataFrame([dict(index=1,question='Q1',A='one',B='two',answer='A'),
                                   dict(index=2,question='Q2',A='one',B='two',answer='B')])
            path=directory/'mcq.xlsx'
            mcq.data.assign(prediction=['A','B']).to_excel(path,index=False)
            result=mcq.evaluate(str(path),model='exact_matching',nproc=1)
            self.assertEqual(float(result['Overall'].iloc[0]),1.)
            mme=ImageYORNDataset.__new__(ImageYORNDataset);mme.dataset_name='MME'
            categories=['OCR','artwork','celebrity','color','count','existence','landmark','position','posters','scene',
                        'code_reasoning','commonsense_reasoning','numerical_calculation','text_translation']
            rows=[dict(index=i*2+j,category=category,image_path=f'{i}.jpg',answer='Yes',prediction='Yes')
                  for i,category in enumerate(categories) for j in range(2)]
            path=directory/'mme.xlsx';pd.DataFrame(rows).to_excel(path,index=False)
            result=mme.evaluate(str(path),model='exact_matching',nproc=1)
            self.assertEqual(float(result['perception'].iloc[0]),2000)
            self.assertEqual(float(result['reasoning'].iloc[0]),800)

    def test_evaluation_scores_normalize_ocr_and_average_excludes_mme(self):
        from small_gptq.evaluation import normalize_result, export_result, read_scores, DEFAULT_BENCHMARKS
        self.assertEqual(normalize_result('OCRBench', {'Final Score': 650})['score'],65)
        self.assertEqual(normalize_result('OCRBench', {'Final Score Norm': 65})['score'],65)
        self.assertEqual(normalize_result('MMStar', [{'split':'none','Overall':.7}])['score'],70)
        with self.assertRaises(ValueError):
            normalize_result('MMStar',[{'split':'a','Overall':.2},{'split':'b','Overall':.8}])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'metrics.json'
            for name in DEFAULT_BENCHMARKS:
                raw={'perception':[1200], 'reasoning':[300]} if name=='MME' else (
                    {'Final Score':500} if name=='OCRBench' else {'Overall':.5})
                export_result(path,'deepseek_vl2_small',name,raw)
            scores=read_scores(path,DEFAULT_BENCHMARKS)
            self.assertEqual(scores['MME'],1500)
            self.assertEqual(scores['MME_perception'],1200)
            self.assertEqual(scores['Avg'],50)
            data=json.loads(path.read_text());del data['deepseek_vl2_small']['OCRBench']
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'Missing evaluation'):
                read_scores(path,DEFAULT_BENCHMARKS)

    def test_all_runs_three_independent_seeds_and_aggregates_outputs(self):
        from small_gptq.cli import main
        from small_gptq.evaluation import DEFAULT_BENCHMARKS, export_result
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            def execute(command, **kwargs):
                self.assertTrue(kwargs['check'])
                stage=command[2]
                seed=int(command[command.index('--seed')+1])
                # The final work-dir override takes precedence in argparse.
                work_index=max(i for i,x in enumerate(command) if x=='--work-dir')
                output=Path(command[work_index+1])
                self.assertEqual(output,work/f'seed_{seed}')
                if stage=='evaluate':
                    for name in DEFAULT_BENCHMARKS:
                        raw={'perception':1200+seed,'reasoning':300} if name=='MME' else (
                            {'Final Score':500+seed*10} if name=='OCRBench' else {'Overall':.5+seed*.01})
                        export_result(output/'evaluation/metrics.json','deepseek_vl2_small',name,raw)
            with patch('small_gptq.cli.subprocess.run',side_effect=execute) as run, redirect_stdout(io.StringIO()):
                main(['all','--model','model','--data','calib.jsonl','--work-dir',str(work)])
            self.assertEqual(run.call_count,15)
            result=json.loads((work/'evaluation_summary.json').read_text())
            self.assertEqual(result['seeds'],[0,1,2])
            self.assertEqual(result['metrics']['Avg']['mean'],51)
            self.assertEqual(result['metrics']['Avg']['std'],1)
            self.assertTrue((work/'evaluation_summary.csv').is_file())
            self.assertIn('OCRBench',result['metrics'])

    def test_calibration_requires_all_paper_sources_and_preserves_128_unique_rows(self):
        from small_gptq.model import select_calibration_rows
        from small_gptq.cli import parser
        families = ['DocVQA', 'InfoVQA', 'ScienceQA', 'TextVQA', 'ChartQA']
        rows = [dict(source=name+'_VAL', id=i*40+j) for i,name in enumerate(families) for j in range(40)]
        got = select_calibration_rows(rows, 128, 0)
        self.assertEqual(len({r['id'] for r in got}), 128)
        self.assertEqual([sum(r['source'].startswith(n+'_') for r in got) for n in families], [26,26,26,25,25])
        self.assertEqual(got, select_calibration_rows(rows, 128, 0))
        with self.assertRaisesRegex(ValueError, 'Missing paper'):
            select_calibration_rows(rows[:120], 128)
        with self.assertRaisesRegex(ValueError, 'outside'):
            select_calibration_rows(rows+[dict(source='POPE')], 128)
        self.assertEqual(parser().parse_args(['bank']).seqlen, 2048)

    def test_calibration_builder_extracts_images_and_rejects_missing_sources(self):
        import importlib.util
        import csv
        import base64
        from PIL import Image
        spec=importlib.util.spec_from_file_location('paper_calib',ROOT/'newest/data/build_multimodal_calib.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        image=io.BytesIO();Image.new('RGB',(2,2)).save(image,format='PNG')
        encoded=base64.b64encode(image.getvalue()).decode()
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            for source in module.SOURCES:
                with (directory/(source+'.tsv')).open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['index','question','answer','image'],delimiter='\t')
                    writer.writeheader()
                    for i in range(3):
                        writer.writerow(dict(index=i,question='What?',answer="['yes', 'y']",image=encoded if i==0 else '0'))
            with redirect_stdout(io.StringIO()):
                result=module.build(directory,directory/'calib.jsonl',10,0)
            self.assertEqual(len(result),10)
            self.assertEqual({r['source'] for r in result},set(module.SOURCES))
            self.assertTrue(all(Path(r['image_path']).is_file() for r in result))
            self.assertTrue(all(r['conversations'][1]['content']=='yes' for r in result))
            (directory/(module.SOURCES[-1]+'.tsv')).unlink()
            with self.assertRaises(FileNotFoundError):
                module.build(directory,directory/'missing.jsonl',10,0)
            self.assertFalse((directory/'missing.jsonl').exists())

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(9)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def bank(self, model):
        directory = self.directory / 'bank'; directory.mkdir()
        manifest = dict(format='owen-small-mcmoe-gptq-v1', group_size=128, modules={})
        self.original = {}
        for index, (name, module) in enumerate(model.named_modules()):
            if '.mlp.experts.' not in name or not isinstance(module, nn.Linear):
                continue
            self.original[name] = module.weight.detach().clone()
            entry = dict(shape=list(module.weight.shape), candidates={})
            for bits in (2, 4):
                # Deliberately synthetic fixed weights, not a fake GPTQ accuracy claim.
                weight = module.weight.detach().clone() * (1 if bits == 4 else .4)
                filename = f'{index}-{bits}.pt'
                torch.save({'weight': weight}, directory / filename)
                entry['candidates'][str(bits)] = filename
            manifest['modules'][name] = entry
        torch.save({}, directory / 'background.pt')
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        return GPTQBank(directory)

    def test_reference_arithmetic_methods_are_structurally_identical(self):
        ref = ast.parse((ROOT / 'pipeline/small_gptq/reference_gptq.py.txt').read_text(encoding='utf-8-sig'))
        ours = ast.parse((ROOT / 'pipeline/small_gptq/backend.py').read_text(encoding='utf-8'))
        def methods(tree):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GPTQ')
            return {n.name: ast.dump(n, include_attributes=False) for n in cls.body if isinstance(n, ast.FunctionDef)}
        a, b = methods(ref), methods(ours)
        for name in ['__init__', 'add_batch', 'static_fasterquant', 'fasterquant']:
            self.assertEqual(a[name], b[name], name)

    def test_candidates_match_reference_codes_and_decoding_for_two_and_four_bits(self):
        for bits in (2, 4):
            original = nn.Linear(8, 3, bias=False)
            before = original.weight.detach().clone()
            layer = copy.deepcopy(original)
            engine = backend.GPTQ(layer, None, '', bits)
            engine.add_batch(torch.randn(2, 5, 8), torch.zeros(2, 5, 3))
            h = engine.H.clone()
            engine.quantizer.configure(bits, perchannel=True, sym=False, mse=False, pack=True)
            with patch('torch.cuda.synchronize'):
                scale, zero, index, _ = engine.fasterquant(groupsize=4, actorder=False)
                got = candidate(original, h, bits, group_size=4)
            self.assertTrue(torch.equal(got['codes'], layer.weight.to(torch.uint8)))
            expected = scale.float()[:, index.long()] * (layer.weight.float() - zero.float()[:, index.long()])
            torch.testing.assert_close(got['weight'], expected, rtol=0, atol=0)
            torch.testing.assert_close(original.weight, before, rtol=0, atol=0)

    def test_uniform_permutation_weights_do_not_collapse_at_large_expert_counts(self):
        for n in (4, 64, 1664):
            p = torch.ones(n, dtype=torch.float64) / n
            permutations = torch.stack([torch.arange(n), torch.arange(n - 1, -1, -1)])
            weights = permutation_weights(permutations, p)
            self.assertTrue(all(abs(w - 1.) < 1e-8 for w in weights))

    def test_importance_weight_ratios_match_analytic_proposal(self):
        p = torch.tensor([.1, .2, .7], dtype=torch.float64)
        permutations = torch.tensor([[0, 1, 2], [2, 1, 0]])
        w = permutation_weights(permutations, p)
        q0 = .1 * (.2 / .9); q1 = .7 * (.2 / .3)
        self.assertAlmostEqual(w[0] / w[1], q1 / q0)

    def test_only_estimator_matches_direct_configuration_values_and_restores_weights(self):
        model = Model().eval(); bank = self.bank(model); bank.bind(model)
        inputs = [(torch.randn(1, 3, 3), torch.tensor([[-100, 2, 4]])),
                  (torch.randn(1, 3, 3), torch.tensor([[-100, -100, 3]]))]
        permutations = torch.tensor([[0, 2, 1, 3], [3, 1, 2, 0]])
        prior = torch.ones(4) / 4
        menus = [menu for menu in enumerate_menus() if menu]
        expected = {f'L{l}.E{e}': {menu_to_key(m): 0. for m in menus} for l in range(2) for e in range(2)}
        def set_expert(gid, menu):
            l, e = divmod(gid, 2)
            for p in ['gate_proj', 'up_proj', 'down_proj']:
                name = f'language.model.layers.{l}.mlp.experts.{e}.{p}'
                module = model.get_submodule(name)
                module.weight.data = bank.weight(name, 4 if p in menu else 2, module, 128)
        with torch.no_grad():
            for order in permutations:
                for gid in range(4): set_expert(gid, frozenset(['gate_proj','up_proj','down_proj']))
                for gid in order.tolist():
                    set_expert(gid, frozenset()); baseline = value(model, inputs)
                    for menu in menus:
                        set_expert(gid, menu)
                        expected[f'L{gid//2}.E{gid%2}'][menu_to_key(menu)] += (value(model, inputs) - baseline) / 2
                    set_expert(gid, frozenset())
            for name, original in self.original.items(): model.get_submodule(name).weight.copy_(original)
        estimator = ConfigOwenEstimator(4, device=torch.device('cpu'), early_stop_threshold=None)
        with redirect_stdout(io.StringIO()):
            got = estimator.estimate_configuration_values(model, inputs, permutations, prior, 2, 0, value, batch_size=2)
        for expert in expected:
            for key in expected[expert]: self.assertAlmostEqual(got[expert][key], expected[expert][key], places=6)
        for name, original in self.original.items(): self.assertTrue(torch.equal(model.get_submodule(name).weight, original))
        self.assertFalse(hasattr(estimator, 'estimate_menu_value'))

    def test_router_mass_sampling_collects_real_gate_output(self):
        model = Model().eval()
        inputs = [(torch.randn(1, 3, 3), torch.tensor([[-100, 2, 4]]))]
        p, orders = profile_and_sample(model, inputs, 3, 7, start_layer=0)
        self.assertEqual(orders.shape, (3, 4)); self.assertAlmostEqual(p.sum().item(), 1.)
        self.assertTrue(torch.isfinite(p).all()); self.assertTrue((p > 0).all())

    def test_exact_dp_matches_bruteforce_and_routed_budget(self):
        model = Model(); bank = self.bank(model)
        # Two experts only, with complete seven-menu tables and interaction effects.
        keep = {n: e for n, e in bank.modules.items() if 'layers.0.' in n}
        bank.modules = keep
        values = {}
        menus = [''] + [menu_to_key(m) for m in enumerate_menus() if m]
        for e in range(2): values[f'L0.E{e}'] = {k: float((i * (e + 3)) % 11 - 2) for i,k in enumerate(menus[1:])}
        for budget in (2., 2.6, 3.1, 4.):
            assignment, summary = solve(values, bank, budget)
            best = -math.inf
            total = sum(math.prod(e['shape']) for e in keep.values())
            for pair in itertools.product(menus, repeat=2):
                cost=0; utility=0
                for e,key in enumerate(pair):
                    selected=set(key.split(',')) if key else set()
                    utility += values[f'L0.E{e}'].get(key, 0.)
                    for p in ['gate_proj','up_proj','down_proj']:
                        cost += math.prod(keep[f'language.model.layers.0.mlp.experts.{e}.{p}']['shape']) * (4 if p in selected else 2)
                if cost <= budget*total + 1e-8: best=max(best,utility)
            self.assertAlmostEqual(summary['total_utility'], best)
            self.assertLessEqual(summary['actual_average_bits'], budget+1e-12)
            self.assertEqual(set(assignment),set(keep))

    def test_assembly_reads_saved_candidates_without_calling_quantizer(self):
        model = Model(); bank = self.bank(model)
        assignment={n: 2 if i%2 else 4 for i,n in enumerate(bank.modules)}
        with patch('small_gptq.build.candidate',side_effect=AssertionError('No online quantization')):
            bank.assemble(model, assignment)
        for name,bits in assignment.items():
            expected=torch.load(bank.path/bank.modules[name]['candidates'][str(bits)],weights_only=True)['weight']
            self.assertTrue(torch.equal(model.get_submodule(name).weight,expected))

    def test_real_gptq_bank_build_and_assembly_match_on_tiny_model(self):
        from small_gptq.build import build_bank
        model = Model().eval(); original = copy.deepcopy(model)
        samples = [dict(inputs_embeds=torch.randn(1, 4, 3), labels=torch.tensor([[-100, 2, 3, 4]]))]
        directory = self.directory / 'generated'
        with patch('torch.cuda.synchronize'), redirect_stdout(io.StringIO()):
            manifest = build_bank(model, None, samples, directory, '.', seqlen=4, quantize_vision=False)
        self.assertEqual(len(manifest['modules']),12)
        bank = GPTQBank(directory)
        bank.assemble(original, {name:4 for name in bank.modules})
        x = samples[0]['inputs_embeds']
        torch.testing.assert_close(model.language(x).logits, original.language(x).logits, rtol=0, atol=0)

    def test_solve_entry_writes_a_complete_routed_assignment_without_cuda(self):
        from small_gptq.cli import main
        bank = self.bank(Model())
        work = self.directory / 'run'; work.mkdir()
        menus = [menu_to_key(m) for m in enumerate_menus() if m]
        values = {f'L{l}.E{e}': {m:float(i) for i,m in enumerate(menus)} for l in range(2) for e in range(2)}
        (work/'configuration_values.json').write_text(json.dumps(values))
        with redirect_stdout(io.StringIO()):
            main(['solve','--bank',str(bank.path),'--work-dir',str(work),'--average-bits','2.6'])
        result=json.loads((work/'assignment.json').read_text())
        summary=json.loads((work/'assignment_summary.json').read_text())
        self.assertEqual(set(result),set(bank.modules))
        self.assertEqual(summary['budget_scope'],'routed_experts')
        self.assertLessEqual(summary['actual_average_bits'],2.6)

    def test_evaluation_parser_accepts_fixed_bank_arguments(self):
        import argparse
        tree=ast.parse((ROOT/'newest/VLMEvalKit/run.py').read_text(encoding='utf-8-sig'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_args')
        namespace={'argparse':argparse}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'evaluation-parser','exec'),namespace)
        with patch.object(sys,'argv',['run.py','--model','deepseek_vl2_small','--data','MMStar','--model_precision','fixed_gptq',
                                      '--gptq_bank','bank','--json_file','assignment.json','--model_device','cuda:0']):
            args=namespace['parse_args']()
        self.assertEqual(args.gptq_bank,'bank')
        self.assertEqual(args.model_precision,'fixed_gptq')

    def test_multimodal_floats_follow_model_dtype_without_casting_token_ids(self):
        from small_gptq.model import to_model_inputs
        model=Model().to(torch.bfloat16)
        sample={'images':torch.ones(1,3,4,4,dtype=torch.float32),
                'input_ids':torch.ones(1,4,dtype=torch.long),'attention_mask':torch.ones(1,4,dtype=torch.bool)}
        got=to_model_inputs(model,sample)
        self.assertEqual(got['images'].dtype,torch.bfloat16)
        self.assertEqual(got['input_ids'].dtype,torch.long)
        self.assertEqual(got['attention_mask'].dtype,torch.bool)

    def test_mla_bypassed_projection_collects_hessian_from_its_normalized_input(self):
        from small_gptq.build import collect
        norm=nn.LayerNorm(4); projection=nn.Linear(4,3,bias=False)
        x=torch.randn(2,5,4)
        def forward():
            # Deliberately bypass projection.forward, as MLA does.
            return torch.nn.functional.linear(norm(x),projection.weight)
        engines=collect({'kv_b':projection},forward,{'kv_b':norm})
        expected=backend.GPTQ(projection,None,'',4)
        expected.add_batch(norm(x),torch.zeros(2,5,3))
        self.assertEqual(engines['kv_b'].nsamples,2)
        torch.testing.assert_close(engines['kv_b'].H,expected.H,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()

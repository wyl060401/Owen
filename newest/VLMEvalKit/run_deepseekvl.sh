
module use /soft/modulefiles/
module load conda
conda activate Deepseek_VL

export LMUData="/lus/grand/projects/datascience/krishnat/home_dir_code/datasets/"

python run.py --model_precision="fp_baseline" --work-dir ./deepseek_vl_tiny_results_fp16_MME --data MME --model "deepseek_vl2_tiny" --mode all
python run.py --model_precision="uniform_quant" --bits=8 --work-dir ./deepseek_vl_tiny_uniform_8_MME --data MME --model "deepseek_vl2_tiny" --mode all

# python run.py --model_precision="fp_baseline" --work-dir ./deepseek_vl_tiny_results_fp16_OCRBench --data OCRBench --model "deepseek_vl2_tiny" --mode all
# python run.py --model_precision="fp_baseline" --work-dir ./deepseek_vl_tiny_results_fp16_MMMU_VAL --data MMMU_VAL --model "deepseek_vl2_tiny" --mode all
# python run.py --model_precision="fp_baseline" --work-dir ./deepseek_vl_tiny_results_fp16_MathVista --data MathVista --model "deepseek_vl2_tiny" --mode all
# python run.py --model_precision="fp_baseline" --work-dir ./deepseek_vl_tiny_results_fp16_OCRVQA --data OCRVQA --model "deepseek_vl2_tiny" --mode all
# python run.py --model_precision="fp_baseline" --work-dir ./deepseek_vl_tiny_results_fp16_ChartVQA --data ChartVQA --model "deepseek_vl2_tiny" --mode all
# python run.py --model_precision="fp_baseline" --work-dir ./deepseek_vl_tiny_results_fp16_AI2D --data AI2D --model "deepseek_vl2_tiny" --mode all
# python run.py --model_precision="fp_baseline" --work-dir ./deepseek_vl_tiny_results_fp16_DoCVQA --data DocVQA --model "deepseek_vl2_tiny" --mode all
# python run.py --model_precision="fp_baseline" --work-dir ./deepseek_vl_tiny_results_fp16_POPE --data POPE --model "deepseek_vl2_tiny" --mode all
# python run.py --model_precision="fp_baseline" --work-dir ./deepseek_vl_tiny_results_fp16_TextVQA_VAL --data TextVQA_VAL --model "deepseek_vl2_tiny" --mode all
# python run.py --model_precision="fp_baseline" --work-dir ./deepseek_vl_tiny_results_fp16_MMMU_TEST --data MMMU_TEST --model "deepseek_vl2_tiny" --mode all
# python run.py --model_precision="uniform_quant" --bits=8 --work-dir ./deepseek_vl_tiny_uniform_8 --data MME --model "deepseek_vl2_tiny" --mode all
# python run.py --model_precision="uniform_quant" --bits=4 --work-dir ./deepseek_vl_tiny_results_uniform_4 --data MME --model "deepseek_vl2_tiny" --mode all
# python run.py --model_precision="uniform_quant" --bits=2 --work-dir ./deepseek_vl_tiny_results_uniform_2 --data MME --model "deepseek_vl2_tiny" --mode all
# python run.py --model_precision="uniform_quant" --bits=4 --work-dir ./deepseek_vl_tiny_results_uniform_4 --data MME --model "deepseek_vl2_tiny" --mode all
# python run.py --model_precision="uniform_quant" --bits=8 --work-dir ./deepseek_vl_tiny_results_uniform_8 --data MME --model "deepseek_vl2_tiny" --mode all
# python run.py --model_precision="fp_baseline" --work-dir ./deepseek_vl_small_results_fp16 --data MME --model "deepseek_vl2_small" --mode all
# python run.py --model_precision="uniform_quant" --bits=2 --work-dir ./deepseek_vl_small_results_uniform_2 --data MME --model "deepseek_vl2_small" --mode all
# python run.py --model_precision="uniform_quant" --bits=4 --work-dir ./deepseek_vl_small_results_uniform_4 --data MME --model "deepseek_vl2_small" --mode all
# python run.py --model_precision="uniform_quant" --bits=8 --work-dir ./deepseek_vl_small_results_uniform_8 --data MME --model "deepseek_vl2_small" --mode all
# python run.py --model_precision="mixed_precision_quant" --bits=8 --work-dir ./deepseek_vl_small_mixed_precision_quant_results_rem_8 --data MME --model "deepseek_vl2_small" --mode all
# python run.py --model_precision="mixed_precision_quant" --bits=4 --work-dir ./deepseek_vl_small_mixed_precision_quant_results_rem_4 --data MME --model "deepseek_vl2_small" --mode all
# python run.py --model_precision="mixed_precision_quant" --bits=8 --work-dir ./deepseek_vl_small_mixed_precision_quant_results_rem_8 --data MME --model "deepseek_vl2_tiny" --mode all
# python run.py --model_precision="mixed_precision_quant" --bits=4 --work-dir ./deepseek_vl_small_mixed_precision_quant_results_rem_4 --data MME --model "deepseek_vl2_tiny" --mode all

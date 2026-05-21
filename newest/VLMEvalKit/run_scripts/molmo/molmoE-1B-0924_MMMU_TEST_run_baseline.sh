#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=25:00:00
#PBS -q preemptable
#PBS -A datascience

module use /soft/modulefiles/
module load conda
conda activate Deepseek_VL

export LMUData="/lus/grand/projects/datascience/krishnat/home_dir_code/datasets/"

cd /lus/grand/projects/datascience/krishnat/home_dir_code/Active_projects/Mixed_precision_MoE/VLMEvalKit/

python run.py --model_precision activation_frequency_profiling --work-dir ./all_results/molmoE-1B-0924_bits_activation_frequency_profiling_activation_frequency_profiling_MMMU_TEST --data MMMU_TEST --model molmoE-1B-0924 --mode all 
python run.py --bits 16 --model_precision fp_baseline --work-dir ./all_results/molmoE-1B-0924_bits_16_fp_baseline_fp_baseline_MMMU_TEST --data MMMU_TEST --model molmoE-1B-0924 --mode all --saveresults 
python run.py --bits 16 --model_precision fp_baseline --work-dir ./all_results/molmoE-1B-0924_bits_16_fp_baseline_fp_baseline_MMMU_TEST --data MMMU_TEST --model molmoE-1B-0924 --mode all --saveresults 
python run.py --bits 8 --model_precision uniform_quant --quant_format auto_gptq --work-dir ./all_results/molmoE-1B-0924_bits_8_uniform_quant_uniform_quant_MMMU_TEST_auto_gptq --data MMMU_TEST --model molmoE-1B-0924 --mode all --saveresults 
python run.py --bits 8 --model_precision uniform_quant --quant_format auto_awq --work-dir ./all_results/molmoE-1B-0924_bits_8_uniform_quant_uniform_quant_MMMU_TEST_auto_awq --data MMMU_TEST --model molmoE-1B-0924 --mode all --saveresults 
python run.py --bits 4 --model_precision uniform_quant --quant_format auto_gptq --work-dir ./all_results/molmoE-1B-0924_bits_4_uniform_quant_uniform_quant_MMMU_TEST_auto_gptq --data MMMU_TEST --model molmoE-1B-0924 --mode all --saveresults 
python run.py --bits 4 --model_precision uniform_quant --quant_format auto_awq --work-dir ./all_results/molmoE-1B-0924_bits_4_uniform_quant_uniform_quant_MMMU_TEST_auto_awq --data MMMU_TEST --model molmoE-1B-0924 --mode all --saveresults 
python run.py --bits 2 --model_precision uniform_quant --quant_format auto_gptq --work-dir ./all_results/molmoE-1B-0924_bits_2_uniform_quant_uniform_quant_MMMU_TEST_auto_gptq --data MMMU_TEST --model molmoE-1B-0924 --mode all --saveresults 
python run.py --bits 2 --model_precision uniform_quant --quant_format auto_awq --work-dir ./all_results/molmoE-1B-0924_bits_2_uniform_quant_uniform_quant_MMMU_TEST_auto_awq --data MMMU_TEST --model molmoE-1B-0924 --mode all --saveresults 

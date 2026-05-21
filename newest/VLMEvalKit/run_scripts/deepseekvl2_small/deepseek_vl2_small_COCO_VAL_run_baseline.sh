#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=15:00:00
#PBS -q preemptable
#PBS -A datascience

module use /soft/modulefiles/
module load conda
conda activate Deepseek_VL

export LMUData="/lus/grand/projects/datascience/krishnat/home_dir_code/datasets/"

cd /lus/grand/projects/datascience/krishnat/home_dir_code/Active_projects/MoE-Mixed-Prec/VLMEvalKit/

python run.py --bits 16 --model_precision fp_baseline --quant_format GPTQ --work-dir ./all_results/deepseek_vl2_small_bits_16_fp_baseline_fp_baseline_COCO_VAL_GPTQ --data COCO_VAL --model deepseek_vl2_small --mode all 
python run.py --bits 16 --model_precision fp_baseline --quant_format AWQ --work-dir ./all_results/deepseek_vl2_small_bits_16_fp_baseline_fp_baseline_COCO_VAL_AWQ --data COCO_VAL --model deepseek_vl2_small --mode all 
python run.py --bits 8 --model_precision uniform_quant --quant_format GPTQ --work-dir ./all_results/deepseek_vl2_small_bits_8_uniform_quant_uniform_quant_COCO_VAL_GPTQ --data COCO_VAL --model deepseek_vl2_small --mode all 
python run.py --bits 8 --model_precision uniform_quant --quant_format AWQ --work-dir ./all_results/deepseek_vl2_small_bits_8_uniform_quant_uniform_quant_COCO_VAL_AWQ --data COCO_VAL --model deepseek_vl2_small --mode all 
python run.py --bits 4 --model_precision uniform_quant --quant_format GPTQ --work-dir ./all_results/deepseek_vl2_small_bits_4_uniform_quant_uniform_quant_COCO_VAL_GPTQ --data COCO_VAL --model deepseek_vl2_small --mode all 
python run.py --bits 4 --model_precision uniform_quant --quant_format AWQ --work-dir ./all_results/deepseek_vl2_small_bits_4_uniform_quant_uniform_quant_COCO_VAL_AWQ --data COCO_VAL --model deepseek_vl2_small --mode all 
python run.py --bits 2 --model_precision uniform_quant --quant_format GPTQ --work-dir ./all_results/deepseek_vl2_small_bits_2_uniform_quant_uniform_quant_COCO_VAL_GPTQ --data COCO_VAL --model deepseek_vl2_small --mode all 
python run.py --bits 2 --model_precision uniform_quant --quant_format AWQ --work-dir ./all_results/deepseek_vl2_small_bits_2_uniform_quant_uniform_quant_COCO_VAL_AWQ --data COCO_VAL --model deepseek_vl2_small --mode all 

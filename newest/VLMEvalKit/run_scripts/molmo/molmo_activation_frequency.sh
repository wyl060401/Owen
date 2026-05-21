#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=15:00:00
#PBS -q preemptable
#PBS -A datascience

module use /soft/modulefiles/
module load conda
conda activate MoE_Mixed_Precision

export LMUData="/lus/grand/projects/datascience/krishnat/home_dir_code/datasets/"

cd /lus/grand/projects/datascience/krishnat/home_dir_code/Active_projects/Mixed_precision_MoE/VLMEvalKit/

# python run.py --model_precision activation_frequency_profiling --model molmoE-1B-0924 --work-dir ./all_results/molmoE-1B_activation_profiling --data AI2D_TEST ChartQA_TEST COCO_VAL DocVQA_TEST MME --mode all

python run.py --model_precision activation_frequency_profiling --work-dir ./all_results/molmoE-1B_activation_profiling --data ChartQA_TEST --model molmoE-1B-0924 --mode all 

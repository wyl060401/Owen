






import os

content = f"""#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=15:00:00
#PBS -q preemptable
#PBS -A datascience

module use /soft/modulefiles/
module load conda
conda activate Deepseek_VL

export LMUData="/lus/grand/projects/datascience/krishnat/home_dir_code/datasets/"

cd /lus/grand/projects/datascience/krishnat/home_dir_code/Active_projects/Mixed_precision_MoE/VLMEvalKit/

"""

run_all_file = False

for model in ["deepseek_vl2_tiny", "deepseek_vl2_small", "deepseek_vl2"]:

    for dataset in [
                    "AI2D_TEST",
                    "ChartQA_TEST",
                    "COCO_VAL",
                    "DocVQA_TEST",
                    "InfoVQA_TEST",
                    "MME",
                    "MMMU_TEST",
                    "OCRVQA_TEST",
                    ]:
        
        output_write_file = f"{model}_{dataset}_run_baseline.sh"

        with open(output_write_file, "w") as file:
            file.write(content)
        
        op = "a" if run_all_file else "w"
        run_all_file = True
        with open("run_all_baseline_files.sh", op) as file:
            file.write(f"chmod +x {output_write_file}")
            file.write("\n")
            file.write(f"qsub {output_write_file}")
            file.write("\n")

            for bits in [16, 8, 4, 2]:
                for quant_format in ["auto_gptq", "auto_awq"]:
                    
                    if bits == 16:
                        model_precision = "fp_baseline"

                    else:
                        model_precision = "uniform_quant"

                    work_dir = f"./all_results/{model}_bits_{bits}_{model_precision}_{model_precision}_{dataset}_{quant_format}"

                    command = f"""python run.py --bits {bits} --model_precision {model_precision} --quant_format {quant_format} --work-dir {work_dir} --data {dataset} --model {model} --mode all 
"""
                    with open(output_write_file, "a") as file:
                        file.write(command)

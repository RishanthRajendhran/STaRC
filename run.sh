#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH -o outputs-%j

export PYTHONPATH=/scratch/general/vast/u1419542/miniconda3/envs/starcEnv/bin/python
source /scratch/general/vast/u1419542/miniconda3/etc/profile.d/conda.sh
conda activate starcEnv

wandb disabled
# mkdir /scratch/general/vast/u1419542/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/u1419542/huggingface_cache"

python3 inference.py -direct -trainPrompts -trainFiles "./datasets/commonsense_qa/promptsDirect.txt"
# python3 inference.py -direct -trainPrompts -trainFiles "./datasets/commonsense_qa/promptsDirect.txt" -rationalize -hintTrainPrompts "./datasets/commonsense_qa/promptsDirectWithHints.txt"
# python3 inference.py -trainPrompts -trainFiles "./datasets/commonsense_qa/prompts.txt" -rationalize -hintTrainPrompts "./datasets/commonsense_qa/promptsWithHints.txt"
# python3 inference.py -trainPrompts -trainFiles "./datasets/commonsense_qa/prompts.txt" 
# deepspeed --num_gpus=1 inference.py -trainPrompts -trainFiles "./datasets/commonsense_qa/prompts.txt" -deepSpeed --do_eval --deepspeed /scratch/general/vast/u0403624/3b-working/0-0/CQA3ds/src/Supervised/deepspeed_config.json
# deepspeed --num_gpus=1 inference.py -deepSpeed -trainPrompts -trainFiles "./datasets/commonsense_qa/prompts.txt"
#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --qos=marasovic-gpulong-np
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=36:00:00
#SBATCH --mem=128GB
#SBATCH -o outputs-%j

ITERATION=8
STOPITERATION=5e-3
NUMSTEPS=40


while getopts 'i:n:s:' opt; do
  case "$opt" in
    i)   ITERATION="$OPTARG"  ;;
    n)   NUMSTEPS="$OPTARG"  ;;
    s)   STOPITERATION="$OPTARG"  ;;
    *) echo "Unexpected option: $1 - this should not happen.";  
       usage; exit 1;;
  esac
done

if (($STOPITERATION == $ITERATION)); then 
    echo "Stop iteration reached. Terminating..."; 
    exit 1; 
fi

export PYTHONPATH=/scratch/general/vast/u1419542/miniconda3/envs/starcEnv/bin/python
source /scratch/general/vast/u1419542/miniconda3/etc/profile.d/conda.sh
conda activate starcEnv

# wandb disabled
# mkdir /scratch/general/vast/u1419542/huggingface_cache
export HF_HOME=/scratch/general/vast/u1419542/huggingface_cache
export HF_DATASETS_CACHE="/scratch/general/vast/u1419542/huggingface_cache"
export TRANSFORMERS_CACHE="/scratch/general/vast/u1419542/huggingface_cache"

PREVITERATION=$((ITERATION-1))
OLDMODELNAME="finetunedGPTJ6B_${PREVITERATION}"
MODELNAME="finetunedGPTJ6B_${ITERATION}"
SAVEMODELNAME="GPTJ6BFineTuned_${ITERATION}"
MODELPATH="./model/${SAVEMODELNAME}"
TRAINFILES="./inferenceOuts/${OLDMODELNAME}/${OLDMODELNAME}_prompts_train_commonsense_qa_correct.json ./inferenceOuts/${OLDMODELNAME}/${OLDMODELNAME}_prompts_train_commonsense_qa_rationalizedCorrect.json"
NEXTITERATION=$((ITERATION+1))
NEXTSTEPS=$(echo "$NUMSTEPS*1.2/1" | bc )

echo "**SCHEDULE JOBS**"
echo "ITERATION : ${ITERATION}"
echo "NUMSTEPS : ${NUMSTEPS}"
echo "STOPITERATION : ${STOPITERATION}"
echo "<<FINETUNE>>"
python3 finetune.py -trainFiles $TRAINFILES -model gptj -size 6b -dataset commonsense_qa -trainPattern .*/.*\.json -log stdout -batchSize 8 -numEpochs $NUMSTEPS -learningRate 1e-6 -savePath ./model/ -saveName $SAVEMODELNAME -maxSteps $NUMSTEPS -trainPrompt ./datasets/commonsense_qa/prompts.txt
if [ $? != 0 ];
then
    echo "exit 1"
fi
echo "<<INFERENCE>> [VALIDATION]"
deepspeed --num_gpus=1 inference.py -deepSpeed -out ./inferenceOuts/ -rationalize -trainPrompts -trainFiles ./datasets/commonsense_qa/prompts.txt -hintTrainPrompts ./datasets/commonsense_qa/promptsWithHints.txt -testFiles validation -model gptj -size 6b -dataset commonsense_qa -maxShots 9 -trainPattern .*/.*\.json -testPattern .*/.*\.json -log stdout -modelPath $MODELPATH -saveAs $MODELNAME
if [ $? != 0 ];
then
    echo "exit 1"
fi
echo "<<INFERENCE>> [TRAIN]"
deepspeed --num_gpus=1 inference.py -deepSpeed -out ./inferenceOuts/ -rationalize -trainPrompts -trainFiles ./datasets/commonsense_qa/prompts.txt -hintTrainPrompts ./datasets/commonsense_qa/promptsWithHints.txt -testFiles train -model gptj -size 6b -dataset commonsense_qa -maxShots 9 -trainPattern .*/.*\.json -testPattern .*/.*\.json -log stdout -modelPath $MODELPATH -saveAs $MODELNAME
if [ $? != 0 ];
then
    echo "exit 1"
fi
echo "<<NEXT>>"
sbatch scheduleJobs.sh -n $NEXTSTEPS -i $NEXTITERATION  -s $STOPITERATION
#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH -o outputs-%j

DEEPSPEED=false
DIRECT=false
FINETUNE=false
INFERENCE=false
ISTRAINDIR=false
ISTESTDIR=false

LOGFILE="stdout"
TRAIN="./datasets/commonsense_qa/prompts.txt"
TEST="validation"
TRAINPATT=".*/.*\\.json"
TESTPATT=".*/.*\\.json"
MODEL="unifiedqa"
MODELSIZE="3b"
DATASET="commonsense_qa"
MODELPATH="allenai/unifiedqa-t5-3b"
BATCHSIZE=8
LEARNINGRATE=5e-3
NUMEPOCHS=1
MODELNAME="UnifiedQA3BFineTuned"
SAVEAS="base"
MAXSTEPS=40
TRAINPROMPT="None"


while getopts 'a:b:c:de:g:j:k:l:m:n:p:q:s:t:u:v:w:xy' opt; do
  case "$opt" in
    a)   LOGFILE="$OPTARG"  ;;
    b)   BATCHSIZE="$OPTARG"  ;;
    c)   LEARNINGRATE="$OPTARG"  ;;
    d)   DEEPSPEED=true   ;;
    e)   NUMEPOCHS="$OPTARG"  ;;
    g)  DATASET="$OPTARG"   ;;
    j)   TRAINPATT="$OPTARG"     ;;
    k)   TESTPATT="$OPTARG"     ;;
    l)  MODELSIZE="$OPTARG"   ;;
    m)   MODEL="$OPTARG"     ;;
    n)   DIRECT=true     ;;
    p)   TRAINPROMPT="$OPTARG"  ;;
    q)   MODELPATH="$OPTARG"     ;;
    s)  MAXSTEPS="$OPTARG"   ;;
    t) TRAIN="$OPTARG" ;;
    u) MODELNAME="$OPTARG" ;;
    v) TEST="$OPTARG" ;;
    w)  SAVEAS="$OPTARG" ;;
    x)   ISTRAINDIR=true     ;;
    y)    ISTESTDIR=true     ;;
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done

export PYTHONPATH=/scratch/general/vast/u1419542/miniconda3/envs/starcEnv/bin/python
source /scratch/general/vast/u1419542/miniconda3/etc/profile.d/conda.sh
conda activate starcEnv

wandb disabled
# mkdir /scratch/general/vast/u1419542/huggingface_cache
export HF_HOME=/scratch/general/vast/u1419542/huggingface_cache
export HF_DATASETS_CACHE="/scratch/general/vast/u1419542/huggingface_cache"
export TRANSFORMERS_CACHE="/scratch/general/vast/u1419542/huggingface_cache"

ADDITIONAL=""
if [ "$ISTRAINDIR" = true ] ; then
    ADDITIONAL="${ADDITIONAL} -isTrainDir"
fi ;

if [ "$ISTESTDIR" = true ] ; then
    ADDITIONAL="${ADDITIONAL} -isTestDir"
fi ;

if [ "$DEEPSPEED" = true ] ; then
    if [ "$DIRECT" = true ] ; then 
        if [ "$ISTRAINDIR" = true ] ; then 
            deepspeed finetune.py -deepSpeed -direct -isTrainDir -trainFiles $TRAIN -model $MODEL -size $MODELSIZE -dataset $DATASET -trainPattern $TRAINPATT -log $LOGFILE -batchSize $BATCHSIZE -numEpochs $NUMEPOCHS -learningRate $LEARNINGRATE -savePath $MODELPATH -saveName $MODELNAME --deepspeed ./deepspeed_config_training.json -maxSteps $MAXSTEPS -trainPrompt $TRAINPROMPT $ADDITIONAL
        else 
            deepspeed finetune.py -deepSpeed -direct -trainFiles $TRAIN -model $MODEL -size $MODELSIZE -dataset $DATASET -trainPattern $TRAINPATT -log $LOGFILE -batchSize $BATCHSIZE -numEpochs $NUMEPOCHS -learningRate $LEARNINGRATE -savePath $MODELPATH -saveName $MODELNAME --deepspeed ./deepspeed_config_training.json -maxSteps $MAXSTEPS -trainPrompt $TRAINPROMPT $ADDITIONAL
        fi  ;
    else 
        if [ "$ISTRAINDIR" = true ] ; then 
            deepspeed finetune.py -deepSpeed -isTrainDir -trainFiles $TRAIN -model $MODEL -size $MODELSIZE -dataset $DATASET -trainPattern $TRAINPATT -log $LOGFILE -batchSize $BATCHSIZE -numEpochs $NUMEPOCHS -learningRate $LEARNINGRATE -savePath $MODELPATH -saveName $MODELNAME  --deepspeed ./deepspeed_config_training.json -maxSteps $MAXSTEPS -trainPrompt $TRAINPROMPT $ADDITIONAL
        else 
            deepspeed finetune.py -deepSpeed -trainFiles $TRAIN -model $MODEL -size $MODELSIZE -dataset $DATASET -trainPattern $TRAINPATT -log $LOGFILE -batchSize $BATCHSIZE -numEpochs $NUMEPOCHS -learningRate $LEARNINGRATE -savePath $MODELPATH -saveName $MODELNAME --deepspeed ./deepspeed_config_training.json -maxSteps $MAXSTEPS -trainPrompt $TRAINPROMPT $ADDITIONAL
        fi  ;
    fi  ;
else 
    if [ "$DIRECT" = true ] ; then 
        if [ "$ISTRAINDIR" = true ] ; then 
            accelerate launch --config_file ds_zero3_cpu.yaml finetune.py -direct -isTrainDir -trainFiles $TRAIN -model $MODEL -size $MODELSIZE -dataset $DATASET -trainPattern $TRAINPATT -log $LOGFILE -batchSize $BATCHSIZE -numEpochs $NUMEPOCHS -learningRate $LEARNINGRATE -savePath $MODELPATH -saveName $MODELNAME -maxSteps $MAXSTEPS -trainPrompt $TRAINPROMPT $ADDITIONAL
        else 
            accelerate launch --config_file ds_zero3_cpu.yaml finetune.py -direct -trainFiles $TRAIN -model $MODEL -size $MODELSIZE -dataset $DATASET -trainPattern $TRAINPATT -log $LOGFILE -batchSize $BATCHSIZE -numEpochs $NUMEPOCHS -learningRate $LEARNINGRATE -savePath $MODELPATH -saveName $MODELNAME -maxSteps $MAXSTEPS -trainPrompt $TRAINPROMPT $ADDITIONAL
        fi  ;
    else 
        if [ "$ISTRAINDIR" = true ] ; then 
            accelerate launch --config_file ds_zero3_cpu.yaml finetune.py -isTrainDir -trainFiles $TRAIN -model $MODEL -size $MODELSIZE -dataset $DATASET -trainPattern $TRAINPATT -log $LOGFILE -batchSize $BATCHSIZE -numEpochs $NUMEPOCHS -learningRate $LEARNINGRATE -savePath $MODELPATH -saveName $MODELNAME -maxSteps $MAXSTEPS -trainPrompt $TRAINPROMPT $ADDITIONAL
        else 
            accelerate launch --config_file ds_zero3_cpu.yaml finetune.py -trainFiles $TRAIN -model $MODEL -size $MODELSIZE -dataset $DATASET -trainPattern $TRAINPATT -log $LOGFILE -batchSize $BATCHSIZE -numEpochs $NUMEPOCHS -learningRate $LEARNINGRATE -savePath $MODELPATH -saveName $MODELNAME -maxSteps $MAXSTEPS -trainPrompt $TRAINPROMPT $ADDITIONAL
        fi  ;
    fi  ;
fi  ;
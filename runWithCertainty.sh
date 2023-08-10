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
#SBATCH --mail-user=u1419542@utah.edu
#SBATCH --mail-type=BEGIN,FAIL,END

DEEPSPEED=false
DIRECT=false
FINETUNE=false
INFERENCE=false
TRAINPROMPTS=false
RATIONALIZE=false
ISTRAINDIR=false
ISTESTDIR=false
ZEROSHOT=false

LOGFILE="stdout"
TRAIN="./datasets/commonsense_qa/prompts.txt"
HINTTRAINPROMPTS="./datasets/commonsense_qa/promptsWithHints.txt"
TEST="validation"
TRAINPATT=".*/.*\\.json"
TESTPATT=".*/.*\\.json"
MODEL="gptj"
MODELSIZE="6b"
DATASET="commonsense_qa"
OUTPUT="./inferenceOuts/"
MAXSHOTS=9
MODELPATH="EleutherAI/gpt-j-6B"
BATCHSIZE=8
LEARNINGRATE=5e-3
NUMEPOCHS=1
MODELNAME="GPTJ6BFineTuned"
SAVEAS=""


while getopts 'a:b:c:de:fg:h:ij:k:l:m:no:pq:rs:t:v:w:xyz' opt; do
  case "$opt" in
    a)   LOGFILE="$OPTARG"  ;;
    b)   BATCHSIZE="$OPTARG"  ;;
    c)   LEARNINGRATE="$OPTARG"  ;;
    d)   DEEPSPEED=true   ;;
    e)   NUMEPOCHS="$OPTARG"  ;;
    f)   FINETUNE=true     ;;
    g)  DATASET="$OPTARG"   ;;
    h)   HINTTRAINPROMPTS="$OPTARG"  ;;
    i)   INFERENCE=true     ;;
    j)   TRAINPATT="$OPTARG"     ;;
    k)   TESTPATT="$OPTARG"     ;;
    l)  MODELSIZE="$OPTARG"   ;;
    m)   MODEL="$OPTARG"     ;;
    n)   DIRECT=true     ;;
    o)   OUTPUT="$OPTARG"     ;;
    p)   TRAINPROMPTS=true  ;;
    q)   MODELPATH="$OPTARG"     ;;
    r)   RATIONALIZE=true  ;;
    s)   MAXSHOTS="$OPTARG"     ;;
    t) TRAIN="$OPTARG" ;;
    u) MODELNAME="$OPTARG" ;;
    v) TEST="$OPTARG" ;;
    w)  SAVEAS="$OPTARG" ;;
    x)   ISTRAINDIR=true     ;;
    y)    ISTESTDIR=true     ;;
    z)   ZEROSHOT=true     ;;
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done

export PYTHONPATH=/scratch/general/vast/u1419542/miniconda3/envs/starcEnv/bin/python
source /scratch/general/vast/u1419542/miniconda3/etc/profile.d/conda.sh
conda activate starcEnv

# wandb disabled
# mkdir /scratch/general/vast/u1419542/huggingface_cache
export HF_HOME=/scratch/general/vast/u1419542/huggingface_cache
export HF_DATASETS_CACHE="/scratch/general/vast/u1419542/huggingface_cache"
export TRANSFORMERS_CACHE="/scratch/general/vast/u1419542/huggingface_cache"

if [ "$INFERENCE" = true ] ; then 
    if [ "$DEEPSPEED" = true ] ; then
        if [ "$TRAINPROMPTS" = true ] ; then
            if [ "$DIRECT" = true ] ; then 
                if [ "$RATIONALIZE" = true ] ; then 
                    if [ "$ZEROSHOT" = true ] ; then
                        deepspeed --num_gpus=1 inferenceWithCertainty.py -zeroShot -rationalize -direct -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    else 
                        deepspeed --num_gpus=1 inferenceWithCertainty.py -rationalize -direct -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    fi  ;
                else 
                    if [ "$ZEROSHOT" = true ] ; then
                        deepspeed --num_gpus=1 inferenceWithCertainty.py -zeroShot -direct -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    else 
                        deepspeed --num_gpus=1 inferenceWithCertainty.py -direct -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    fi  ;
                fi  ;
            else 
                if [ "$RATIONALIZE" = true ] ; then 
                    if [ "$ZEROSHOT" = true ] ; then
                        deepspeed --num_gpus=1 inferenceWithCertainty.py -zeroShot -rationalize -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    else 
                        deepspeed --num_gpus=1 inferenceWithCertainty.py -rationalize -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    fi  ;
                else 
                    if [ "$ZEROSHOT" = true ] ; then
                        deepspeed --num_gpus=1 inferenceWithCertainty.py -zeroShot -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    else 
                        deepspeed --num_gpus=1 inferenceWithCertainty.py -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    fi  ;
                fi  ;
            fi  ;
        else 
            if [ "$DIRECT" = true ] ; then 
                if [ "$RATIONALIZE" = true ] ; then 
                    if [ "$ZEROSHOT" = true ] ; then
                        deepspeed --num_gpus=1 inferenceWithCertainty.py -zeroShot -rationalize -direct -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    else 
                        deepspeed --num_gpus=1 inferenceWithCertainty.py -rationalize -direct -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    fi  ;
                else 
                    if [ "$ZEROSHOT" = true ] ; then
                        deepspeed --num_gpus=1 inferenceWithCertainty.py -zeroShot -direct -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    else 
                        deepspeed --num_gpus=1 inferenceWithCertainty.py -direct -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    fi  ;
                fi  ;
            else 
                if [ "$RATIONALIZE" = true ] ; then 
                    if [ "$ZEROSHOT" = true ] ; then
                        deepspeed --num_gpus=1 inferenceWithCertainty.py -zeroShot -rationalize -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    else 
                        deepspeed --num_gpus=1 inferenceWithCertainty.py -rationalize -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    fi  ;
                else 
                    if [ "$ZEROSHOT" = true ] ; then
                        deepspeed --num_gpus=1 inferenceWithCertainty.py -zeroShot -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    else 
                        deepspeed --num_gpus=1 inferenceWithCertainty.py -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    fi  ;
                fi  ;
            fi  ;
        fi  ;
    else 
        if [ "$TRAINPROMPTS" = true ] ; then
            if [ "$DIRECT" = true ] ; then 
                if [ "$RATIONALIZE" = true ] ; then 
                    if [ "$ZEROSHOT" = true ] ; then
                        python3 inferenceWithCertainty.py -zeroShot -rationalize -direct -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    else 
                        python3 inferenceWithCertainty.py -rationalize -direct -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    fi  ;
                else 
                    if [ "$ZEROSHOT" = true ] ; then
                        python3 inferenceWithCertainty.py -zeroShot -direct -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    else 
                        python3 inferenceWithCertainty.py -direct -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    fi  ;
                fi  ;
            else 
                if [ "$RATIONALIZE" = true ] ; then 
                    if [ "$ZEROSHOT" = true ] ; then
                        python3 inferenceWithCertainty.py -zeroShot -rationalize -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    else 
                        python3 inferenceWithCertainty.py -rationalize -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    fi  ;
                else 
                    if [ "$ZEROSHOT" = true ] ; then
                        python3 inferenceWithCertainty.py -zeroShot -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    else 
                        python3 inferenceWithCertainty.py -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    fi  ;
                fi  ;
            fi  ;
        else 
            if [ "$DIRECT" = true ] ; then 
                if [ "$RATIONALIZE" = true ] ; then 
                    if [ "$ZEROSHOT" = true ] ; then
                        python3 inferenceWithCertainty.py -zeroShot -rationalize -direct -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    else 
                        python3 inferenceWithCertainty.py -rationalize -direct -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    fi  ;
                else 
                    if [ "$ZEROSHOT" = true ] ; then
                        python3 inferenceWithCertainty.py -zeroShot -direct -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    else 
                        python3 inferenceWithCertainty.py -direct -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    fi  ;
                fi  ;
            else 
                if [ "$RATIONALIZE" = true ] ; then 
                    if [ "$ZEROSHOT" = true ] ; then
                        python3 inferenceWithCertainty.py -zeroShot -rationalize -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    else 
                        python3 inferenceWithCertainty.py -rationalize -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    fi  ;
                else 
                    if [ "$ZEROSHOT" = true ] ; then
                        python3 inferenceWithCertainty.py -zeroShot -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    else 
                        python3 inferenceWithCertainty.py -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS
                    fi  ;
                fi  ;
            fi  ;
        fi  ;
    fi  ;
else
    if [ "$FINETUNE" = true ] ; then
        if [ "$DEEPSPEED" = true ] ; then
            if [ "$DIRECT" = true ] ; then 
                if [ "$ISTRAINDIR" = true ] ; then 
                    deepspeed finetuneWithCertainty.py -direct -isTrainDir -trainFiles $TRAIN -model $MODEL -size $MODELSIZE -dataset $DATASET -trainPattern $TRAINPATT -log $LOGFILE -batchSize $BATCHSIZE -numEpochs $NUMEPOCHS -learningRate $LEARNINGRATE -savePath $MODELPATH -saveName $MODELNAME --deepspeed ./deepspeed_config_training.json
                else 
                    deepspeed finetuneWithCertainty.py -direct -trainFiles $TRAIN -model $MODEL -size $MODELSIZE -dataset $DATASET -trainPattern $TRAINPATT -log $LOGFILE -batchSize $BATCHSIZE -numEpochs $NUMEPOCHS -learningRate $LEARNINGRATE -savePath $MODELPATH -saveName $MODELNAME --deepspeed ./deepspeed_config_training.json
                fi  ;
            else 
                if [ "$ISTRAINDIR" = true ] ; then 
                    deepspeed finetuneWithCertainty.py -isTrainDir -trainFiles $TRAIN -model $MODEL -size $MODELSIZE -dataset $DATASET -trainPattern $TRAINPATT -log $LOGFILE -batchSize $BATCHSIZE -numEpochs $NUMEPOCHS -learningRate $LEARNINGRATE -savePath $MODELPATH -saveName $MODELNAME  --deepspeed ./deepspeed_config_training.json
                else 
                    deepspeed finetuneWithCertainty.py -trainFiles $TRAIN -model $MODEL -size $MODELSIZE -dataset $DATASET -trainPattern $TRAINPATT -log $LOGFILE -batchSize $BATCHSIZE -numEpochs $NUMEPOCHS -learningRate $LEARNINGRATE -savePath $MODELPATH -saveName $MODELNAME --deepspeed ./deepspeed_config_training.json
                fi  ;
            fi  ;
        else 
            if [ "$DIRECT" = true ] ; then 
                if [ "$ISTRAINDIR" = true ] ; then 
                    python3 finetuneWithCertainty.py -direct -isTrainDir -trainFiles $TRAIN -model $MODEL -size $MODELSIZE -dataset $DATASET -trainPattern $TRAINPATT -log $LOGFILE -batchSize $BATCHSIZE -numEpochs $NUMEPOCHS -learningRate $LEARNINGRATE -savePath $MODELPATH -saveName $MODELNAME
                else 
                    python3 finetuneWithCertainty.py -direct -trainFiles $TRAIN -model $MODEL -size $MODELSIZE -dataset $DATASET -trainPattern $TRAINPATT -log $LOGFILE -batchSize $BATCHSIZE -numEpochs $NUMEPOCHS -learningRate $LEARNINGRATE -savePath $MODELPATH -saveName $MODELNAME
                fi  ;
            else 
                if [ "$ISTRAINDIR" = true ] ; then 
                    python3 finetuneWithCertainty.py -isTrainDir -trainFiles $TRAIN -model $MODEL -size $MODELSIZE -dataset $DATASET -trainPattern $TRAINPATT -log $LOGFILE -batchSize $BATCHSIZE -numEpochs $NUMEPOCHS -learningRate $LEARNINGRATE -savePath $MODELPATH -saveName $MODELNAME
                else 
                    python3 finetuneWithCertainty.py -trainFiles $TRAIN -model $MODEL -size $MODELSIZE -dataset $DATASET -trainPattern $TRAINPATT -log $LOGFILE -batchSize $BATCHSIZE -numEpochs $NUMEPOCHS -learningRate $LEARNINGRATE -savePath $MODELPATH -saveName $MODELNAME
                fi  ;
            fi  ;
        fi  ;
    else 
        echo "Select either Inference/Finetune!"
    fi  ;
fi  ;
    

# python3 inferenceWithCertainty.py -direct -trainPrompts -trainFiles "./datasets/commonsense_qa/promptsDirect.txt"
# python3 inferenceWithCertainty.py -direct -trainPrompts -trainFiles "./datasets/commonsense_qa/promptsDirect.txt" -rationalize -hintTrainPrompts "./datasets/commonsense_qa/promptsDirectWithHints.txt"
# python3 inferenceWithCertainty.py -trainPrompts -trainFiles "./datasets/commonsense_qa/prompts.txt" -rationalize -hintTrainPrompts "./datasets/commonsense_qa/promptsWithHints.txt"
# python3 inferenceWithCertainty.py -trainPrompts -trainFiles "./datasets/commonsense_qa/prompts.txt" 
# deepspeed --num_gpus=1 inferenceWithCertainty.py -trainPrompts -trainFiles "./datasets/commonsense_qa/prompts.txt" --do_eval --deepspeed /scratch/general/vast/u0403624/3b-working/0-0/CQA3ds/src/Supervised/deepspeed_config.json
# deepspeed --num_gpus=1 inferenceWithCertainty.py -trainPrompts -trainFiles "./datasets/commonsense_qa/prompts.txt"
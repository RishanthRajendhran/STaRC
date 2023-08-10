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

DEEPSPEED=false
DIRECT=false
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
MODEL="unifiedqa"
MODELSIZE="3b"
DATASET="commonsense_qa"
OUTPUT="./inferenceOuts/"
MAXSHOTS=9
MODELPATH="allenai/unifiedqa-t5-3b"
SAVEAS="base"


while getopts 'a:dg:h:j:k:l:m:no:pq:rs:t:v:w:xyz' opt; do
  case "$opt" in
    a)   LOGFILE="$OPTARG"  ;;
    d)   DEEPSPEED=true   ;;
    g)  DATASET="$OPTARG"   ;;
    h)   HINTTRAINPROMPTS="$OPTARG"  ;;
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

ADDITIONAL=""
if [ "$ISTRAINDIR" = true ] ; then
    ADDITIONAL="${ADDITIONAL} -isTrainDir"
fi ;

if [ "$ISTESTDIR" = true ] ; then
    ADDITIONAL="${ADDITIONAL} -isTestDir"
fi ;

if [ "$DEEPSPEED" = true ] ; then
    if [ "$TRAINPROMPTS" = true ] ; then
        if [ "$DIRECT" = true ] ; then 
            if [ "$RATIONALIZE" = true ] ; then 
                if [ "$ZEROSHOT" = true ] ; then
                    deepspeed --num_gpus=1 inference.py -out $OUTPUT -deepSpeed -zeroShot -rationalize -direct -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                else 
                    deepspeed --num_gpus=1 inference.py -out $OUTPUT -deepSpeed -rationalize -direct -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                fi  ;
            else 
                if [ "$ZEROSHOT" = true ] ; then
                    deepspeed --num_gpus=1 inference.py -out $OUTPUT -deepSpeed -zeroShot -direct -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                else 
                    deepspeed --num_gpus=1 inference.py -out $OUTPUT -deepSpeed -direct -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                fi  ;
            fi  ;
        else 
            if [ "$RATIONALIZE" = true ] ; then 
                if [ "$ZEROSHOT" = true ] ; then
                    deepspeed --num_gpus=1 inference.py -out $OUTPUT -deepSpeed -zeroShot -rationalize -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                else 
                    deepspeed --num_gpus=1 inference.py -out $OUTPUT -deepSpeed -rationalize -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                fi  ;
            else 
                if [ "$ZEROSHOT" = true ] ; then
                    deepspeed --num_gpus=1 inference.py -out $OUTPUT -deepSpeed -zeroShot -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                else 
                    deepspeed --num_gpus=1 inference.py -out $OUTPUT -deepSpeed -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                fi  ;
            fi  ;
        fi  ;
    else 
        if [ "$DIRECT" = true ] ; then 
            if [ "$RATIONALIZE" = true ] ; then 
                if [ "$ZEROSHOT" = true ] ; then
                    deepspeed --num_gpus=1 inference.py -out $OUTPUT -deepSpeed -zeroShot -rationalize -direct -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                else 
                    deepspeed --num_gpus=1 inference.py -out $OUTPUT -deepSpeed -rationalize -direct -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                fi  ;
            else 
                if [ "$ZEROSHOT" = true ] ; then
                    deepspeed --num_gpus=1 inference.py -out $OUTPUT -deepSpeed -zeroShot -direct -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                else 
                    deepspeed --num_gpus=1 inference.py -out $OUTPUT -deepSpeed -direct -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                fi  ;
            fi  ;
        else 
            if [ "$RATIONALIZE" = true ] ; then 
                if [ "$ZEROSHOT" = true ] ; then
                    deepspeed --num_gpus=1 inference.py -out $OUTPUT -deepSpeed -zeroShot -rationalize -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                else 
                    deepspeed --num_gpus=1 inference.py -out $OUTPUT -deepSpeed -rationalize -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                fi  ;
            else 
                if [ "$ZEROSHOT" = true ] ; then
                    deepspeed --num_gpus=1 inference.py -out $OUTPUT -deepSpeed -zeroShot -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                else 
                    deepspeed --num_gpus=1 inference.py -out $OUTPUT -deepSpeed -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                fi  ;
            fi  ;
        fi  ;
    fi  ;
else 
    if [ "$TRAINPROMPTS" = true ] ; then
        if [ "$DIRECT" = true ] ; then 
            if [ "$RATIONALIZE" = true ] ; then 
                if [ "$ZEROSHOT" = true ] ; then
                    python3 inference.py -out $OUTPUT -zeroShot -rationalize -direct -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                else 
                    python3 inference.py -out $OUTPUT -rationalize -direct -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                fi  ;
            else 
                if [ "$ZEROSHOT" = true ] ; then
                    python3 inference.py -out $OUTPUT -zeroShot -direct -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                else 
                    python3 inference.py -out $OUTPUT -direct -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                fi  ;
            fi  ;
        else 
            if [ "$RATIONALIZE" = true ] ; then 
                if [ "$ZEROSHOT" = true ] ; then
                    python3 inference.py -out $OUTPUT -zeroShot -rationalize -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                else 
                    python3 inference.py -out $OUTPUT -rationalize -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                fi  ;
            else 
                if [ "$ZEROSHOT" = true ] ; then
                    python3 inference.py -out $OUTPUT -zeroShot -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                else 
                    python3 inference.py -out $OUTPUT -trainPrompts -trainFiles $TRAIN -hintTrainPrompts $HINTTRAINPROMPTS -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                fi  ;
            fi  ;
        fi  ;
    else 
        if [ "$DIRECT" = true ] ; then 
            if [ "$RATIONALIZE" = true ] ; then 
                if [ "$ZEROSHOT" = true ] ; then
                    python3 inference.py -out $OUTPUT -zeroShot -rationalize -direct -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                else 
                    python3 inference.py -out $OUTPUT -rationalize -direct -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                fi  ;
            else 
                if [ "$ZEROSHOT" = true ] ; then
                    python3 inference.py -out $OUTPUT -zeroShot -direct -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                else 
                    python3 inference.py -out $OUTPUT -direct -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                fi  ;
            fi  ;
        else 
            if [ "$RATIONALIZE" = true ] ; then 
                if [ "$ZEROSHOT" = true ] ; then
                    python3 inference.py -out $OUTPUT -zeroShot -rationalize -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                else 
                    python3 inference.py -out $OUTPUT -rationalize -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                fi  ;
            else 
                if [ "$ZEROSHOT" = true ] ; then
                    python3 inference.py -out $OUTPUT -zeroShot -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                else 
                    python3 inference.py -out $OUTPUT -trainFiles $TRAIN -testFiles $TEST -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPattern $TRAINPATT -testPattern $TESTPATT -log $LOGFILE -modelPath $MODELPATH -saveAs $SAVEAS $ADDITIONAL
                fi  ;
            fi  ;
        fi  ;
    fi  ;
fi  ;
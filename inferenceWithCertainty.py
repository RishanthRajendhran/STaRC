#STaRC - Self-Taught Reasoner with Certainty
import argparse
import wandb
import logging
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import json
import deepspeed
from deepspeed.ops.transformer.inference import DeepSpeedTransformerInference
import json
from tqdm import tqdm
import regex as re
import os
import glob
from os.path import exists
from pathlib import Path

MODEL_TEMPERATURE=0.1
MODEL_TOP_P=0.9
MODEL_TOP_K=0
MODEL_MAX_NEW_TOKENS=256
MODEL_DO_SAMPLE=True
MODEL_REPETITION_PENALTY=1.0

supportedModels = ["gptj"]
supportedSizes = ["6b"]
supportedDatasets = ["commonsense_qa"]
supportedHFDatasets = ["commonsense_qa"]

logging.basicConfig(filemode="w",level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument(
    "-log",
    type=str,
    help="Path to file to print logging information",
    default=None
)

parser.add_argument(
    "-model",
    choices=supportedModels,
    help="Name of HuggingFace model to use",
    default="gptj"
)

parser.add_argument(
    "-modelPath",
    type=str,
    help="Path to (finetuned) model to use",
    default=None
)

parser.add_argument(
    "-size",
    choices=supportedSizes,
    help="Size of HuggingFace model to use",
    default="6b"
)

parser.add_argument(
    "-dataset",
    choices=supportedDatasets,
    help="Name of HuggingFace dataset to use",
    default="commonsense_qa"
)

parser.add_argument(
    "-direct",
    action="store_true",
    help="Boolean flag to enable direct prompting"
)

parser.add_argument(
    "-maxShots",
    type=int,
    help="Maximum no. of shots to use in few-shot setting",
    default=9
)

parser.add_argument(
    "-zeroShot",
    action="store_true",
    help="Boolean flag to enable zero shot evaluation"
)

parser.add_argument(
    "-trainFiles",
    nargs="+",
    help="List of paths to json/txt files containing training data/prompts",
    default=["train"]    
)

parser.add_argument(
    "-trainPrompts",
    action="store_true",
    help="Boolean flag to indicate that -trainFiles points to files containing train prompts"  
)

parser.add_argument(
    "-hintTrainPrompts",
    nargs="+",
    help="List of paths to txt files containing training prompts with hints",
)

parser.add_argument(
    "-testFiles",
    nargs="+",
    help="List of paths to json files containing test data",
    default=["validation"]    
)

parser.add_argument(
    "-isTrainDir",
    action="store_true",
    help="Booleaan flag to indicate if the -trainFiles input is a directory path",
)

parser.add_argument(
    "-isTestDir",
    action="store_true",
    help="Booleaan flag to indicate if the -testFiles input is a directory path",
)

parser.add_argument(
    "-trainPattern",
    help="RegEx pattern for json/txt file names in the train directory that need to be used"
)

parser.add_argument(
    "-testPattern",
    help="RegEx pattern for json file names in the test directory that need to be merged"
)

parser.add_argument(
    "-deepSpeed",
    action="store_true",
    help="Boolean flag to indicate execution through deepspeed"
)

parser.add_argument(
    "-out",
    help="Path to directory where outputs are to be saved",
    default="./inferenceOuts/"
)

parser.add_argument(
    "-saveAs",
    help="Prefix to add to output files",
    default=""
)

parser.add_argument(
    "-rationalize",
    action="store_true",
    help="Boolean flag to enable rationalization"
)

#Arguments for DeepSpeed
parser.add_argument(
    "--local_rank", 
    type=int, 
    help="[DEEPSPEED ARGUMENT]",
    default=0
)

parser.add_argument(
    "--do_eval",
    action="store_true",
    help="[DEEPSPEED ARGUMENT] Boolean flag to enable inference mode"
)

parser.add_argument(
    "--deepspeed", 
    help="[DEEPSPEED ARGUMENT] Path to deepspeed configuration"
)

#---------------------------------------------------------------------------
def _generateIndividualPrompt(instance, dataset, model, direct=False, rationalize=False, isTest=False):
    if dataset not in supportedDatasets:
        raise ValueError(f"{dataset} not supported!")
    
    prompt = ""
    
    #commonsense_qa on HuggingFace
    # {
    #     "id": (string),
    #     "question": (string),
    #     "choices": {
    #         "labels": [(string),...],
    #         "text": [(string),...]
    #     },
    #     "answerKey": (string)
    # }
    if dataset == "commonsense_qa":
        if not direct and not isTest: 
            raise ValueError("Only direct prompting supported with commonsense_qa dataset on HuggingFace!")
        prompt += "Q: " + instance["question"] + "\nAnswer Choices:\n"
        for c, t in zip(instance["choices"]["label"], instance["choices"]["text"]):
            prompt += "({}) {}".format(c.lower(), t.lower())
            if rationalize:
                if c == instance["answerKey"]:
                    prompt += " (CORRECT)" 
            prompt += "\n"
        prompt += "A: "
        if not isTest: 
            prompt += "({}).\n\n".format(instance["answerKey"].lower())
    return prompt

#---------------------------------------------------------------------------
def _generatePrompt(data, dataset, model, maxShots, direct=False, rationalize=False, isTest=False):
    prompts = []

    if dataset not in supportedDatasets:
        raise ValueError(f"{dataset} not supported!")

    for index, instance in enumerate(data):
        if index >= maxShots:
            break 
        prompts.append(_generateIndividualPrompt(instance, dataset, model, direct, rationalize, isTest))
    
    return prompts
#---------------------------------------------------------------------------
def generateTrainPrompt(data, dataset, model, maxShots, direct, rationalize=False):
    return "".join(_generatePrompt(data, dataset, model, maxShots, direct, rationalize, False))
#---------------------------------------------------------------------------
def generateTestPrompt(instance, dataset, model, maxShots, direct, rationalize=False):
    return "".join(_generatePrompt([instance], dataset, model, maxShots, direct, rationalize, True))
#---------------------------------------------------------------------------
def extractAnswer(answer, direct=False):
    if not direct:
        searchPattern = "answer is .*."
    else: 
        searchPattern = "\([a-z]\)."
    matchedSpan = re.search(searchPattern, answer)
    if matchedSpan == None:
        logging.warning(f"Could not extract answer from {answer}!")
        return None
        # raise RuntimeError(f"Could not extract answer from {answer}!")
    extractedAnswer = answer[matchedSpan.start():matchedSpan.end()].strip()
    answerPattern = "\([a-z]\)."
    matchedAnswer = re.findall(answerPattern, extractedAnswer)
    if len(matchedAnswer)==0:
        logging.warning(f"Could not extract answer from {extractedAnswer}!")
        return None
        # raise RuntimeError(f"Could not extract answer from {extractedAnswer}!")
    matchedAnswer = matchedAnswer[-1][1]
    extractedAnswer = {
        "answer":matchedAnswer.strip(),
    }
    if not direct:
        rationale = answer[:matchedSpan.start()]
        rationalePattern = "[.]"
        matchedRationale = re.split(rationalePattern, rationale)
        if len(matchedRationale):
            rationale = ".".join(matchedRationale[:-1])+"."
        extractedAnswer.update({
            "rationale":rationale.strip(), 
        })
    return extractedAnswer
#---------------------------------------------------------------------------
def processArguments(args):
    config = {
        "logFile": args.log,
        "model": args.model,
        "modelPath": args.modelPath,
        "size": args.size,
        "dataset": args.dataset,
        "direct": args.direct,
        "maxShots": args.maxShots,
        "zeroShot": args.zeroShot,
        "trainFiles": args.trainFiles,
        "trainPrompts": args.trainPrompts,
        "isTrainDir": args.isTrainDir,
        "trainPattern": args.trainPattern,
        "testFiles": args.testFiles,
        "isTestDir": args.isTestDir,
        "testPattern": args.testPattern,
        "deepSpeed": args.deepSpeed,
        "outPath": args.out,
        "saveAs": args.saveAs,
        "rationalize": args.rationalize,
        "hintTrainPrompts": args.hintTrainPrompts,
    }

    if config["logFile"]:
        if config["logFile"]!="stdout" and config["logFile"].endswith(".txt"):
            logging.basicConfig(filename=config["logFile"], filemode='w', level=logging.INFO)
        elif config["logFile"]=="stdout":
            logging.basicConfig(filemode='w', level=logging.INFO)
        else: 
            raise ValueError("Invalid log file {}!".format(config["logFile"]))
    else: 
        logging.basicConfig(filemode='w', level=logging.INFO)

    if config["direct"] and config["rationalize"]:
        raise ValueError("Cannot perform rationalization in direct prompting mode!")
    if config["rationalize"] and config["trainPrompts"]:
        if not config["hintTrainPrompts"]:
            raise ValueError("Hint train prompts must be provided with the hintTrainPrompts flag when these flags are set: trainPrompts, rationalize")
        if len(config["trainFiles"]) != len(config["hintTrainPrompts"]):
            raise ValueError("Every train file must have a hinted version when these flags are set: trainPrompts, rationalize")

    if config["isTrainDir"]:
        jsonDirName = config["trainFiles"][0]
        jsonPattern = os.path.join(jsonDirName, '*.json')
        config["trainFiles"] = glob.glob(jsonPattern)
        if config["trainPattern"]:
            try: 
                re.compile(config["trainPattern"])
            except: 
                raise ValueError("{} is not a valid regular expression!".format(config["trainPattern"]))
            config["trainFiles"] = [tf for tf in config["trainFiles"] if re.match(config["trainPattern"], tf)]
            if len(config["trainFiles"]) == 0:
                raise RuntimeError("{} did not match any file!".format(config["trainPattern"]))

    if config["isTestDir"]:
        jsonDirName = config["testFiles"][0]
        jsonPattern = os.path.join(jsonDirName, '*.json')
        config["testFiles"] = glob.glob(jsonPattern)
        if config["testPattern"]:
            try: 
                re.compile(config["testPattern"])
            except: 
                raise ValueError("{} is not a valid regular expression!".format(config["testPattern"]))
            config["testFiles"] = [tf for tf in config["testFiles"] if re.match(config["testPattern"], tf)]
            if len(config["testFiles"]) == 0:
                raise RuntimeError("{} did not match any file!".format(config["testPattern"]))
    
    #Check if file exists
    for trainFile in config["trainFiles"]:
        if not trainFile.endswith(".json") and not trainFile.endswith(".txt"):
            logging.warning(f"Train File '{trainFile}' is not a json/txt file. Not checking if file exists. Ignore this warning if this is expected behaviour.")
            continue
        file_exists = exists(trainFile)
        if not file_exists:
            raise ValueError(f"{trainFile} is an invalid train file path!")
        path = Path(trainFile)
        if not path.is_file():
            raise ValueError(f"{trainFile} is not a (train) file!")
    #Check if file exists
    for testFile in config["testFiles"]:
        if not testFile.endswith(".json"):
            logging.warning(f"Test File '{testFile}' is not a json file. Not checking if file exists. Ignore this warning if this is expected behaviour.")
            continue
        file_exists = exists(testFile)
        if not file_exists:
            raise ValueError(f"{testFile} is an invalid test file path!")
        path = Path(testFile)
        if not path.is_file():
            raise ValueError(f"{testFile} is not a (test) file!")
        
    return config
#---------------------------------------------------------------------------
def infer(model, tokenizer, prompt, generationConfig={}, deepSpeed=False, dsModel=None):
    tokenizedInput = tokenizer(prompt, return_tensors="pt")
    inputIDs = tokenizedInput.input_ids.to(device=model.device)
    attentionMask = tokenizedInput.attention_mask.to(device=model.device)

    if deepSpeed:
        if not dsModel:
            raise RuntimeError(f"dsModel not passed to infer()!")
        genTokens = dsModel.generate(
            input_ids=inputIDs,
            attention_mask=attentionMask,
            max_new_tokens=MODEL_MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            **generationConfig,
        )
    else:
        genTokens = model.generate(
            input_ids=inputIDs,
            attention_mask=attentionMask,
            max_new_tokens=MODEL_MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            **generationConfig,
        )
    logging.info(f"genTokens: {genTokens}")
    scores = genTokens.scores  
    outputIDs = genTokens.sequences[:, len(inputIDs[0]):]
    logging.info(f"Scores:  {scores}")
    logging.info(f"Shape of Scores:  {scores[0].shape}")
    logging.info(f"Length of Scores:  {len(scores)}")
    for i in range(len(scores)):
        logging.info("{}: {}".format(i, scores[i][scores[i]>0]))
    probs = torch.stack(scores, dim=1).softmax(-1) 
    logging.info(f"Probs: {probs}")
    genProbs = torch.gather(probs, 2, outputIDs[:, :, None]).squeeze(-1)
    logging.info(f"GenProbs: {genProbs}")
    exit(0)
    outputIDs = genTokens[0, len(inputIDs[0]):]
    genText = tokenizer.decode(outputIDs)
    return genText 
#---------------------------------------------------------------------------
def main():
    args = parser.parse_args()

    wandb.init(
        project="STaRC",
        config = processArguments(args)
    )

    config = wandb.config

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if config.model == "gptj":
        if config.size == "6b":
            modelID = "EleutherAI/gpt-j-6B"
            if config.modelPath and config.modelPath!=modelID: #Finetuned model
                model = AutoModelForCausalLM.from_pretrained(config.modelPath, load_in_8bit=True)
                generationConfig = {}
            else:   #Pretrained model
                model = AutoModelForCausalLM.from_pretrained(modelID, load_in_8bit=True)
                generationConfig = {
                    "do_sample":MODEL_DO_SAMPLE,
                    "temperature":MODEL_TEMPERATURE,
                    "top_p":MODEL_TOP_P,
                    "top_k":MODEL_TOP_K,
                    "repetition_penalty":MODEL_REPETITION_PENALTY,
                }
            tokenizer = AutoTokenizer.from_pretrained(modelID)
        else: 
            raise ValueError("Only {} size(s) supported!".format("/".join(supportedSizes)))
    else: 
        raise ValueError("Only {} model(s) supported!".format("/".join(supportedModels)))
    
    if config.deepSpeed:
        # model.to(device=args.local_rank)
        dsModel = deepspeed.init_inference(
            model=model,
            mp_size=1,
            # dtype=torch.half,
            replace_method="auto",
            replace_with_kernel_inject=True,
        )
        assert isinstance(dsModel.module.transformer.h[0], DeepSpeedTransformerInference) == True, "Model not sucessfully initalized"
    else: 
        dsModel = None
        # model.to(device=device)
    
    if config.dataset not in supportedDatasets:
        raise ValueError("Only {} dataset(s) supported!".format("/".join(supportedDatasets)))
    
    if config.dataset in supportedHFDatasets:
        dataset = load_dataset(config.dataset)

    logging.info(f"Model: {config.model}-{config.size}")
    logging.info(f"Model Path: {config.modelPath}")
    logging.info(f"Dataset: {config.dataset}")
    if config.rationalize:
        logging.info("Performing rationalization")
    else: 
        logging.info("Not performing rationalization")
    if config.deepSpeed:
        logging.info(f"Inference using DeepSpeed")
    else:
        logging.info(f"Inference without using DeepSpeed")
    
    for trainInd, trainFile in enumerate(tqdm(config.trainFiles, desc="Train File")):
        if not config.zeroShot:
            if trainFile.endswith(".json"):
                with open(trainFile, "r") as f:
                    trainData = json.load(f)
            elif config.trainPrompts:
                if trainFile.endswith(".txt"):
                    with open(trainFile, "r") as f:
                        trainPrompt = f.read()
                else: 
                    raise ValueError(f"{trainFile} prompt file does not have .txt extension!")
                if config.rationalize:
                    hintTrainFile = config.hintTrainPrompts[trainInd]
                    if hintTrainFile.endswith(".txt"):
                        with open(hintTrainFile, "r") as f:
                            rationalizedTrainPrompt = f.read()
                    else: 
                        raise ValueError(f"{hintTrainFile} prompt file with hints does not have .txt extension!")
            elif config.dataset in supportedHFDatasets:
                trainData = list(dataset[trainFile].select(np.random.choice(len(dataset[trainFile]), config.maxShots)))
            else: 
                raise ValueError(f"Neither is {config.dataset} on HugginFace nor has path to files containing training data been provided!")
            if not config.trainPrompts:
                trainPrompt = generateTrainPrompt(trainData, config.dataset, config.model, config.maxShots, config.direct, False)
                rationalizedTrainPrompt = generateTrainPrompt(trainData, config.dataset, config.model, config.maxShots, config.direct, True)

        for testInd, testFile in enumerate(tqdm(config.testFiles, desc="Test File")):
            if config.dataset in supportedHFDatasets:
                testData = list(dataset[testFile])
            else:
                raise ValueError("Only HuggingFace datasets supported for testing!")
            outputs = []
            correctPreds = []
            wrongPreds = []
            rationalizedCorrectPreds = [] 
            rationalizedWrongPreds = []
            accuracyScore = 0
            rationalizedAccuracyScore = 0
            for testInstance in tqdm(testData, desc="Test Instance"):
                testPrompt = generateTestPrompt(testInstance, config.dataset, config.model, config.maxShots, config.direct, False)

                if not config.zeroShot:
                    finalPrompt = trainPrompt + testPrompt
                else:
                    finalPrompt = testPrompt         

                genText = infer(model, tokenizer, finalPrompt, generationConfig, config.deepSpeed, dsModel)

                extractedAnswer = extractAnswer(genText, config.direct)
                if extractedAnswer == None:
                    continue
                prediction = extractedAnswer["answer"]
                testInstance.update({
                    "output": genText,
                    "prediction": prediction,
                })

                if not config.direct:
                    rationale = extractedAnswer["rationale"]
                    testInstance.update({
                        "rationale": rationale,
                    })

                logging.info(f"Prompt:\n{finalPrompt}")
                if not config.direct:
                    logging.info(f"Rationale: {rationale}")
                logging.info(f"Prediction: {prediction}")
                logging.info("Answer: {}".format(testInstance["answerKey"].lower()))
                logging.info("Score: {}".format(prediction.lower() == testInstance["answerKey"].lower()))
                logging.info("-"*25)
                
                outputs.append(testInstance)
                if prediction.lower() == testInstance["answerKey"].lower():
                    accuracyScore += 1
                    correctPreds.append(testInstance)
                else: 
                    wrongPreds.append(testInstance)
                    #Rationalize
                    if config.rationalize:
                        rationalizedTestPrompt = generateTestPrompt(testInstance, config.dataset, config.model, config.maxShots, config.direct, True)

                        if not config.zeroShot:
                            rationalizedFinalPrompt = rationalizedTrainPrompt + rationalizedTestPrompt
                        else:
                            rationalizedFinalPrompt = rationalizedTestPrompt 
                        genText = infer(model, tokenizer, rationalizedFinalPrompt, generationConfig, config.deepSpeed, dsModel)
                        extractedAnswer = extractAnswer(genText, config.direct)
                        if extractedAnswer == None:
                            continue
                        prediction = extractedAnswer["answer"]
                        testInstance.update({
                            "output": genText,
                            "prediction": prediction,
                        })

                        if not config.direct:
                            rationale = extractedAnswer["rationale"]
                            testInstance.update({
                                "rationale": rationale,
                            })

                        logging.info("Performing rationalization...")
                        logging.info(f"Rationalized Prompt:\n{rationalizedFinalPrompt}")
                        if not config.direct:
                            logging.info(f"Rationalized Rationale: {rationale}")
                        logging.info(f"Rationalized Prediction: {prediction}")
                        logging.info("Rationalized Answer: {}".format(testInstance["answerKey"].lower()))
                        logging.info("Rationalized Score: {}".format(prediction.lower() == testInstance["answerKey"].lower()))
                        logging.info("-"*25)
                        
                        outputs.append(testInstance)
                        if prediction.lower() == testInstance["answerKey"].lower():
                            rationalizedAccuracyScore += 1
                            rationalizedCorrectPreds.append(testInstance)
                        else: 
                            rationalizedWrongPreds.append(testInstance)
                logging.info("*"*50)
            logging.info("Accuracy: {:0.2f}% ({}/{})".format((accuracyScore/len(testData))*100, accuracyScore, len(testData)))
            if config.rationalize:
                logging.info("Rationalization Accuracy: {:0.2f}% ({}/{})".format((rationalizedAccuracyScore/(len(rationalizedCorrectPreds)+len(rationalizedWrongPreds)))*100, rationalizedAccuracyScore, (len(rationalizedCorrectPreds)+len(rationalizedWrongPreds))))
            if not config.outPath.endswith("/"):
                config.outPath += "/"
            with open(f'{config.outPath}{config.saveAs}_{trainFile.split("/")[-1].split(".")[0]}_{testFile.split("/")[-1].split(".")[0]}_{config.dataset}.json', 'w') as fout:
                json.dump(outputs , fout)
            with open(f'{config.outPath}{config.saveAs}_{trainFile.split("/")[-1].split(".")[0]}_{testFile.split("/")[-1].split(".")[0]}_{config.dataset}_correct.json', 'w') as fout:
                json.dump(correctPreds , fout)
            with open(f'{config.outPath}{config.saveAs}_{trainFile.split("/")[-1].split(".")[0]}_{testFile.split("/")[-1].split(".")[0]}_{config.dataset}_wrong.json', 'w') as fout:
                json.dump(wrongPreds , fout)
            with open(f'{config.outPath}{config.saveAs}_{trainFile.split("/")[-1].split(".")[0]}_{testFile.split("/")[-1].split(".")[0]}_{config.dataset}_rationalizedCorrect.json', 'w') as fout:
                json.dump(rationalizedCorrectPreds , fout)
            with open(f'{config.outPath}{config.saveAs}_{trainFile.split("/")[-1].split(".")[0]}_{testFile.split("/")[-1].split(".")[0]}_{config.dataset}_rationalizedWrong.json', 'w') as fout:
                json.dump(rationalizedWrongPreds , fout)
            
    wandb.finish()
#---------------------------------------------------------------------------

if __name__=="__main__":
    main()
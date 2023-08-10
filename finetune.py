#STaRC - Self-Taught Reasoner with Certainty
import argparse
import wandb
import logging
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, AdamW, get_scheduler, T5ForConditionalGeneration, T5Tokenizer
from torch.optim import Adam
from datasets import Dataset, load_dataset
import numpy as np
import json
import deepspeed
import json
from tqdm import tqdm
import regex as re
import os
import glob
from os.path import exists
from pathlib import Path
from torch.utils.data import DataLoader
import bitsandbytes as bnb
import math
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
import math
from torch.autograd import Variable

MODEL_TEMPERATURE=0.1
MODEL_TOP_P=0.9
MODEL_TOP_K=0
MODEL_MAX_NEW_TOKENS=128
MODEL_DO_SAMPLE=True
MODEL_REPETITION_PENALTY=1.0
MAXLENGTH=512

supportedModels = ["gptj", "unifiedqa"]
supportedSizes = {
    "gptj": ["6b"],
    "unifiedqa": ["3b"],
}
# supportedDatasets = ["commonsense_qa", "gsm8k", "arithmetic"]
supportedDatasets = ["commonsense_qa", "gsm8k"]
supportedHFDatasets = ["commonsense_qa", "gsm8k"]

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
    "-size",
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
    "-trainFiles",
    nargs="+",
    help="List of paths to json files containing finetuning data",
    default=["train"]    
)

parser.add_argument(
    "-isTrainDir",
    action="store_true",
    help="Booleaan flag to indicate if the -trainFiles input is a directory path",
)

parser.add_argument(
    "-trainPrompt",
    help="Path to file containing few-shot prompt to include with every training instance",
    default="None"  
)

parser.add_argument(
    "-trainPattern",
    help="RegEx pattern for json file names in the train directory that need to be used"
)

parser.add_argument(
    "-numEpochs",
    type=int,
    help="No. of epochs to finetune for",
    default=1
)

parser.add_argument(
    "-batchSize",
    type=int,
    help="Training Batchsize",
    default=8
)

parser.add_argument(
    "-learningRate",
    type=float,
    help="Learning rate for training",
    default=3e-5,
)

parser.add_argument(
    "-savePath",
    type=str,
    help="Path to save model after every epoch of finetuning",
    default="./model/"
)

parser.add_argument(
    "-saveName",
    type=str,
    help="Name to save model as after every epoch of finetuning",
)

parser.add_argument(
    "-maxSteps",
    type=int,
    help="Maximum number of optimization steps allowed",
    default=1
)

parser.add_argument(
    "-evaluate",
    type=int,
    help="When set, model is evaluated on test files, if provided, after every n epochs where n is the value passed to this argument",
    default="-1"
)

parser.add_argument(
    "-testFiles",
    nargs="+",
    help="List of paths to json files containing test data",
    default=["validation"]    
)

parser.add_argument(
    "-isTestDir",
    action="store_true",
    help="Booleaan flag to indicate if the -testFiles input is a directory path",
)

parser.add_argument(
    "-deepSpeed",
    action="store_true",
    help="Boolean flag to indicate execution through deepspeed"
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
class DatasetTokenizer():
    def __init__(self, modelName, tokenizer, dataset, direct=False, trainPrompt="s"):
        self.modelName = modelName
        self.tokenizer = tokenizer
        if dataset not in supportedDatasets:
            raise ValueError(f"{dataset} not supported!")
        self.dataset = dataset
        self.direct = direct
        self.trainPrompt = trainPrompt

    def _generateIndividualPrompt(self, instance): 
        #commonsense_qa on HuggingFace
        # {
        #     "id": (string),
        #     "question": (string),
        #     "choices": {
        #         "labels": [(string),...],
        #         "text": [(string),...]
        #     },
        #     "rationale": (string),
        #     "answerKey": (string)
        # }
        if self.modelName == "gptj":
            if self.dataset == "commonsense_qa":
                prompt = self.trainPrompt
                prompt += "Q: " + instance["question"] + "\nAnswer Choices:\n"
                corrAns = ""
                for c, t in zip(instance["choices"]["label"], instance["choices"]["text"]):
                    prompt += "({}) {}".format(c.lower(), t.lower())
                    prompt += "\n"
                    if c.lower() == instance["answerKey"].lower():
                        corrAns = t
                prompt += "A: "
                if self.direct: 
                    prompt += "({}).\n\n".format(instance["answerKey"].lower())
                else:
                    prompt += "{} Therefore, the answer is {} ({}).\n\n".format(instance["rationale"], corrAns.lower(), instance["answerKey"].lower())
            #gsm8k on HuggingFace
            # {
            #     "question": (string),
            #     "answer": (string)
            # }
            elif self.dataset == "gsm8k": 
                prompt = self.trainPrompt
                prompt += "Q: " + instance["question"] 
                extractedAnswer = extractAnswer(instance["answer"], self.dataset, self.direct)
                prompt += "\nA: "
                if self.direct: 
                    prompt += extractedAnswer["answer"]
                else:
                    prompt += instance["answer"]
            else: 
                raise NotImplementedError(f"Prompt generation not yet implemented for {self.dataset}")
            return prompt
        elif self.modelName == "unifiedqa":
            if self.dataset == "commonsense_qa":
                prompt = self.trainPrompt
                label = ""
                prompt += "Q: " + instance["question"] + "\nAnswer Choices:\n"
                corrAns = ""
                for c, t in zip(instance["choices"]["label"], instance["choices"]["text"]):
                    prompt += "({}) {}".format(c.lower(), t.lower())
                    prompt += "\n"
                    if c.lower() == instance["answerKey"].lower():
                        corrAns = t
                prompt += "A: "
                if self.direct: 
                    label += "({}).\n\n".format(instance["answerKey"].lower())
                else:
                    label += "{} Therefore, the answer is {} ({}).\n\n".format(instance["rationale"], corrAns.lower(), instance["answerKey"].lower())
            #gsm8k on HuggingFace
            # {
            #     "question": (string),
            #     "answer": (string)
            # }
            elif self.dataset == "gsm8k": 
                prompt = self.trainPrompt
                label = ""
                prompt += "Q: " + instance["question"] 
                extractedAnswer = extractAnswer(instance["answer"], self.dataset, self.direct)
                prompt += "\nA: "
                if self.direct: 
                    label += extractedAnswer["answer"]
                else:
                    label += instance["answer"]
            else: 
                raise NotImplementedError(f"Prompt generation not yet implemented for {self.dataset}")
            return prompt, label
        else: 
            raise ValueError("{} model not supported!".format(self.modelName))

    def tokenize(self, instances):
        if self.modelName == "gptj":
            prompt = self._generateIndividualPrompt(instances)
            tokenizedInput = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=MAXLENGTH)
            return tokenizedInput
        elif self.modelName == "unifiedqa":
            prompt, label = self._generateIndividualPrompt(instances)
            tokenizedInput = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=MAXLENGTH)
            tokenizedLabels = self.tokenizer(label, return_tensors="pt", truncation=True, padding="max_length", max_length=MAXLENGTH).input_ids
            tokenizedInput.update({
                "labels": torch.squeeze(tokenizedLabels),
                "input_ids": torch.squeeze(tokenizedInput.input_ids),
                "attention_mask": torch.squeeze(tokenizedInput.attention_mask),
            })
            return tokenizedInput
        else: 
            raise ValueError("{} model not supported!".format(self.modelName))
#---------------------------------------------------------------------------
def extractAnswer(answer, dataset, direct=False):
    if dataset not in supportedDatasets:
        raise ValueError(f"{dataset} not supported!")
    if dataset == "commonsense_qa":
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
    elif dataset == "gsm8k":
        if not direct:
            searchPattern = "\n#### [0-9]+"
        else: 
            searchPattern = "[0-9]+"
        matchedSpan = re.search(searchPattern, answer)
        if matchedSpan == None:
            logging.warning(f"Could not extract answer from {answer}!")
            return None
            # raise RuntimeError(f"Could not extract answer from {answer}!")
        extractedAnswer = answer[matchedSpan.start():matchedSpan.end()].strip()
        if not direct:
            matchedAnswer = re.sub("#","",extractedAnswer).strip()
        else:
            matchedAnswer = extractedAnswer.strip()
        extractedAnswer = {
            "answer":matchedAnswer.strip(),
        }
        if not direct:
            rationale = answer[:matchedSpan.start()]
            extractedAnswer.update({
                "rationale":rationale.strip(), 
            })
    return extractedAnswer
#---------------------------------------------------------------------------
def processArguments(args):
    config = {
        "logFile": args.log,
        "model": args.model,
        "size": args.size,
        "dataset": args.dataset,
        "direct": args.direct,
        "trainFiles": args.trainFiles,
        "isTrainDir": args.isTrainDir,
        "trainPrompt": args.trainPrompt,
        "evaluate": args.evaluate,
        "testFiles": args.testFiles,
        "isTestDir": args.isTestDir,
        "trainPattern": args.trainPattern,
        "deepSpeed": args.deepSpeed,
        "deepspeed": args.deepspeed,
        "numEpochs": args.numEpochs,
        "batchSize": args.batchSize,
        "learningRate": args.learningRate,
        "saveModelPath": args.savePath,
        "maxSteps": args.maxSteps,
    }

    if config["maxSteps"] <= 0:
        logging.warning("maxSteps cannot be non-positive!")
        config["maxSteps"] = -1

    if config["numEpochs"] < 1:
        logging.warning("numEpochs cannot be non-positive!")
        config["numEpochs"] = 1

    if config["evaluate"]!=-1:
        if not config["evaluate"] > 0:
            raise ValueError("Value to argument: evaluate has to be positive!")
        if config.deepSpeed:
            raise NotImplementedError("Cannot perform evaluation when training with deepspeed!")

    if args.saveName:
        config.update({
            "saveModelName": args.saveName
        })
    else: 
        config.update({
            "saveModelName": f"{config.model}{config.modelSize}finetuned"
        })

    if config["logFile"]:
        if config["logFile"]!="stdout" and config["logFile"].endswith(".txt"):
            logging.basicConfig(filename=config["logFile"], filemode='w', level=logging.INFO)
        elif config["logFile"]=="stdout":
            logging.basicConfig(filemode='w', level=logging.INFO)
        elif config["logFile"]=="none":
            logging.basicConfig(filemode='w', level=logging.ERROR)
        else: 
            raise ValueError("Invalid log file {}!".format(config["logFile"]))
    else: 
        logging.basicConfig(filemode='w', level=logging.ERROR)

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
        if not trainFile.endswith(".json"):
            logging.warning(f"Train File '{trainFile}' is not a json file. Not checking if file exists. Ignore this warning if this is expected behaviour.")
            continue
        file_exists = exists(trainFile)
        if not file_exists:
            raise ValueError(f"{trainFile} is an invalid train file path!")
        path = Path(trainFile)
        if not path.is_file():
            raise ValueError(f"{trainFile} is not a (train) file!")

    #Check if file exists
    for testFile in config["testFiles"]:
        if not testFile.endswith(".json") and not testFile.endswith(".jsonl"):
            logging.warning(f"Test File '{testFile}' is not a json/jsonl file. Not checking if file exists. Ignore this warning if this is expected behaviour.")
            continue
        file_exists = exists(testFile)
        if not file_exists:
            raise ValueError(f"{testFile} is an invalid test file path!")
        path = Path(testFile)
        if not path.is_file():
            raise ValueError(f"{testFile} is not a (test) file!")

    if config["trainPrompt"] != "None": 
        if not config["trainPrompt"].endswith(".txt"):
            raise ValueError("Train prompt '{}' is not a txt file!".format(config["trainPrompt"]))
        file_exists = exists(config["trainPrompt"])
        if not file_exists:
            raise ValueError("{} is an invalid train prompt file path!".format(config["trainPrompt"]))
        path = Path(config["trainPrompt"])
        if not path.is_file():
            raise ValueError("{} is not a (train prompt) file!".format(config["trainPrompt"]))
        
    return config
#---------------------------------------------------------------------------
def main():
    args = parser.parse_args()

    wandb.init(
        project="STaRC",
        config = processArguments(args),
        allow_val_change=True
    )

    config = wandb.config

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if config.model == "gptj":
        if config.size == "6b":
            loraConfig = LoraConfig(
                r=8, 
                lora_alpha=32, 
                lora_dropout=0.05, 
                bias="none", 
                task_type="CAUSAL_LM"
            )
            modelID = "EleutherAI/gpt-j-6B"
            print("Using pretrained model and tokenizer from {} on HuggingFace".format(modelID))
            model = AutoModelForCausalLM.from_pretrained(modelID, load_in_8bit=True, device_map="auto")
            model.gradient_checkpointing_enable()
            model = prepare_model_for_int8_training(model)
            model = get_peft_model(model, loraConfig)
            for p in model.parameters():
                p = p.contiguous()
            tokenizer = AutoTokenizer.from_pretrained(modelID)
            tokenizer.pad_token = tokenizer.eos_token
        else: 
            raise ValueError("Only {} size(s) supported!".format("/".join(supportedSizes)))
    elif config.model == "unifiedqa":
        if config.size == "3b":
            loraConfig = LoraConfig(
                r=8, 
                lora_alpha=32, 
                lora_dropout=0.05, 
                bias="none", 
                task_type="SEQ_2_SEQ_LM"
            )
            modelID = "allenai/unifiedqa-t5-3b"
            print("Using pretrained model and tokenizer from {} on HuggingFace".format(modelID))
            model = T5ForConditionalGeneration.from_pretrained(modelID, device_map="auto")
            model.gradient_checkpointing_enable()
            model = prepare_model_for_int8_training(model)
            model = get_peft_model(model, loraConfig)
            for p in model.parameters():
                p = p.contiguous()
            tokenizer = T5Tokenizer.from_pretrained(modelID)
            tokenizer.pad_token = tokenizer.eos_token
        else: 
            raise ValueError("Only {} size(s) supported!".format("/".join(supportedSizes[config.model])))
    else: 
        raise ValueError("Only {} model(s) supported!".format("/".join(supportedModels)))
    
    if config.dataset not in supportedDatasets:
        raise ValueError("Only {} dataset(s) supported!".format("/".join(supportedDatasets)))
    
    if config.dataset in supportedHFDatasets:
        if config.dataset == "gsm8k":
            dataset = load_dataset(config.dataset, "main")
        else:
            dataset = load_dataset(config.dataset)
    
    trainData = []
    for trainFile in config.trainFiles:
        if trainFile.endswith(".json"):
            with open(trainFile, "r") as f:
                trainData.extend(json.load(f))
        elif config.dataset in supportedHFDatasets:
            trainData.extend(list(dataset[trainFile]))
        else: 
            raise ValueError(f"Neither is {config.dataset} on HuggingFace nor has path to json files containing training data been provided!")
    logging.info("No. of examples used for finetuning: {}".format(len(trainData)))

    trainDS = Dataset.from_list(trainData)
    
    if config.evaluate!=-1 and config.testFiles:
        testData = []
        for testFile in config.testFiles:
            if testFile.endswith(".json"):
                with open(testFile, "r") as f:
                    testData.extend(json.load(f))
            elif config.dataset in supportedHFDatasets:
                testData.extend(list(dataset[testFile]))
            else: 
                raise ValueError(f"Neither is {config.dataset} on HuggingFace nor has path to json files containing test data been provided!")
        testDS = Dataset.from_list(testData)

    trainPrompt=""
    if config.trainPrompt!="None":
        with open(config.trainPrompt, "r") as f:
            trainPrompt = f.read()

    if config.dataset == "commonsense_qa":
        tokenizedTrainDS = trainDS.map(DatasetTokenizer(config.model, tokenizer, config.dataset, config.direct, trainPrompt).tokenize, batched=False, remove_columns=trainDS.column_names)
        if config.evaluate!=-1 and config.testFiles:
            testColsToRemove = list(testDS.column_names).remove("answerKey")
            tokenizedTestDS = testDS.map(DatasetTokenizer(config.model, tokenizer, config.dataset, config.direct, trainPrompt).tokenize, batched=False, remove_columns=testColsToRemove)
    elif config.dataset == "gsm8k":
        tokenizedTrainDS = trainDS.map(DatasetTokenizer(config.model, tokenizer, config.dataset, config.direct, trainPrompt).tokenize, batched=False, remove_columns=trainDS.column_names)
        if config.evaluate!=-1 and config.testFiles:
            testColsToRemove = list(testDS.column_names).remove("answer")
            tokenizedTestDS = testDS.map(DatasetTokenizer(config.model, tokenizer, config.dataset, config.direct, trainPrompt).tokenize, batched=False, remove_columns=testColsToRemove)
    else: 
        raise NotImplementedError("Support for {} not yet implemented!".format(config.dataset))
    
    if config.model == "gptj":
        dataCollator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif config.model == "unifiedqa":
        dataCollator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    else: 
        raise NotImplementedError("Data collator for {} not specified!".format(config.model))
    
    tokenizedTrainDS.set_format("torch")
    trainDataLoader = DataLoader(
        tokenizedTrainDS, 
        shuffle=True, 
        batch_size=config.batchSize, 
        collate_fn=dataCollator
    )

    if config.evaluate!=-1 and config.testFiles:
        tokenizedTestDS.set_format("torch")
        testDataLoader = DataLoader(
            tokenizedTestDS, 
            shuffle=True, 
            batch_size=config.batchSize, 
            collate_fn=dataCollator
        )

    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=config.learningRate)

    numTrainingSteps = config.numEpochs * len(trainDataLoader)
    if config.maxSteps == -1:
        config.update({
                "maxSteps": numTrainingSteps
            }, 
            allow_val_change=True
        )
    elif config.maxSteps > 0:
        config.update({
                "numEpochs": math.ceil(config.maxSteps/len(trainDataLoader))
            },
            allow_val_change=True
        )
    else: 
        raise ValueError(f"Maximum no. of steps (maxSteps) has to be positive!")

    lrScheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=numTrainingSteps,
    )

    if config.deepSpeed:
        modelEngine, modelOptimizer, _, _ = deepspeed.initialize(
            args={
                "zero_allow_untested_optimizer": True,
            },
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            lr_scheduler=lrScheduler,
            collate_fn=dataCollator,
            config=config.deepspeed,
        )

    progressBar = tqdm(range(numTrainingSteps))

    bestLoss = np.inf
    numSteps = 0
    for epoch in tqdm(range(config.numEpochs),desc="Epoch"):
        model.train()
        batchInd = 0
        avgLoss = 0
        for batch in tqdm(trainDataLoader, desc="Batch"):
            numSteps += 1
            batchInd += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            if config.deepSpeed:
                loss = modelEngine(batch)
                modelEngine.backward(loss)
                modelEngine.step()
            else:
                outputs = model(**batch)
                loss = outputs.loss 
                loss = Variable(loss, requires_grad = True)
                loss.backward()
                optimizer.step()
                lrScheduler.step()
                optimizer.zero_grad()
            avgLoss += loss.item()
            progressBar.update(1)
            #Update
            avgLoss /= batchInd
            logging.info(f"Epoch {epoch+1}/{config.numEpochs}, Step {numSteps}/{config.maxSteps}: Loss = {avgLoss}")
            wandb.log({
                "loss": avgLoss
            })
            if avgLoss < bestLoss:
                bestLoss = avgLoss
                model.save_pretrained(f"{config.saveModelPath}{config.saveModelName}", from_pt=True) 
                tokenizer.save_pretrained(f"{config.saveModelPath}{config.saveModelName}", from_pt=True)
            if config.evaluate!=-1 and numSteps%config.evaluate==0:
                model.eval()
                with torch.no_grad():
                    for testBatch in tqdm(testDataLoader, desc="Batch"):
                        testBatch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in testBatch.items()}
                        testOutputs = model(**testBatch)
                        testLogits = torch.squeeze(testOutputs.logits)
                        testOutputIDs = testLogits.argmax(-1)
                        testGenText = tokenizer.batch_decode(testOutputIDs)
                        testInpText = tokenizer.batch_decode(torch.squeeze(testBatch["input_ids"]))
                        for inp, out in zip(inpText, genText):
                            extractedAnswer = extractAnswer(instance["answer"], self.dataset, self.direct)
                            raise NotImplementedError("Evaluation: In progress...")
                model.train()
            if math.isclose(avgLoss, 0):
                break
            if numSteps >= config.maxSteps:
                break
        if numSteps >= config.maxSteps:
                break
        logging.info("*"*50)
    wandb.finish()
#---------------------------------------------------------------------------
if __name__=="__main__":
    main()

# ###
# #Randomly print some samples
# if torch.randn(1)>0.75:
#     logits = torch.squeeze(outputs.logits)
#     outputIDs = logits.argmax(-1)
#     genText = tokenizer.batch_decode(outputIDs)
#     inpText = tokenizer.batch_decode(torch.squeeze(batch["input_ids"]))
#     for inp, out in zip(inpText, genText):
#         logging.info("Epoch {}/{}:".format(epoch, config.numEpochs))
#         logging.info(f"Input:\n{inp}")
#         logging.info(f"Output:\n{out}")
#         logging.info("-"*25)
# ###
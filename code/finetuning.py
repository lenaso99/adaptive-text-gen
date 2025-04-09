import os
import pandas as pd
import torch

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

load_dotenv("../.env")
hf_token = os.getenv("HUGGINGFACE_TOKEN")

model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
tokenizer_name_or_path = "meta-llama/Llama-2-7b-chat-hf"

# Load model & tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="cuda", token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

# load data
eng_train = pd.read_xlsx("data/eng_train.xlsx")
eng_test = pd.read_xlsx("data/eng_test.xlsx")
ger_train = pd.read_xlsx("data/ger_train.xlsx")
ger_test = pd.read_xlsx("data/ger_test.xlsx")

level_col = "normalized_level"
text_col = "text"

def tokenize_data(batch):
    inputs = tokenizer(batch["input"], truncation=True, padding="max_length", max_length=128)
    labels = tokenizer([str(x) for x in batch["target"]], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs

# english training
eng_train_tokenized = Dataset.from_pandas(eng_train).map(...)
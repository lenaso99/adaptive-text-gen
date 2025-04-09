import os
import pandas as pd
import torch

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
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
df_eng = pd.read_xlsx("data/eng_data_with_prompt.xlsx")
df_ger = pd.read_xlsx("data/ger_data_with_prompt.xlsx")

eng_train, eng_test = train_test_split(df_eng, test_size=0.2, random_state=42, stratify=df_eng["normalized_level"])
ger_train, ger_test = train_test_split(df_ger, test_size=0.2, random_state=42, stratify=df_ger["normalized_level"])

#TODO

def tokenize_data(batch):
    inputs = tokenizer(batch["input"], truncation=True, padding="max_length", max_length=128)
    labels = tokenizer([str(x) for x in batch["target"]], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs

# english training
eng_train_tokenized = Dataset.from_pandas(eng_train).map(...)
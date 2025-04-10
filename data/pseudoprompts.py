import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv("../.env")
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# load datasets
eng_data = pd.read_excel("../data/eng_data.xlsx")
ger_data = pd.read_excel("../data/ger_data.xlsx")

# load model...
model = "meta-llama/Llama-2-13b-chat-hf"
pipe = pipeline("text-generation", model=model, trust_remote_code=True, token=hf_token)

# generate responses with pipeline
eng_responses = []
for _, row in eng_data.iterrows():
    text = row["text"]
    prompt = f"<INST>Given the following text, please generate a prompt that it could be a response for:\n\"{text}\"</INST>"

    result = pipe(prompt, max_new_tokens=50)
    generated = result.split("</INST>")[-1].strip()

    eng_responses.append(generated)
eng_data["prompt"] = eng_responses
eng_data.to_excel("../data/eng_data_with_prompt.xlsx", index=False)

ger_responses = []
for _, row in ger_data.iterrows():
    text = row["text"]
    prompt = f"<INST>Erfinde anhand des folgenden Textes eine Prompt, für die er eine Antwort sein könnte:\n\"{text}\"</INST>"

    result = pipe(prompt, max_new_tokens=50)
    generated = result.split("</INST>")[-1].strip()

    ger_responses.append(generated)
ger_data["prompt"] = ger_responses
ger_data.to_excel("../data/ger_data_with_prompt.xlsx", index=False)
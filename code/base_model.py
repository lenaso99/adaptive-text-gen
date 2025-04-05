from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
import torch
import transformers
import re
import os

'''
Experimenting with base models, no finetuning or prompting.
'''

# Remove token from code before uploading to GitHub; this is to get access to the models
hf_token = "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"

# Load model of choice
## LLama 2 variants
model_id = "meta-llama/Llama-2-7b-chat-hf"
#model_id = "meta-llama/Llama-2-13b-chat-hf"

## possibly: Mistral models

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda", token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

def clean_model_response(response):
    '''
    With some models (e.g. the Llama models I used here), the model includes the prompt
    in the response. This function cleans the response by removing the prompt and any other
    unwanted characters.
    '''

    cleaned_response = re.sub(r"\"\"\".*?\"\"\"", "", response, flags=re.DOTALL) # response is wrapped in \\\
    cleaned_response = cleaned_response.lstrip() # remove leading white spaces
    cleaned_response = re.sub(r'^[\W_]+', '', cleaned_response) # remove leading non-alphanumeric characters
    return cleaned_response

def chat_with_model(prompt, length):
    prompt = "\"\"\"" + prompt + "\"\"\""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to('cuda')
    output = model.generate(input_ids, max_new_tokens=length, num_beams=4, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response_clean = clean_model_response(response)
    return response_clean

if __name__ == "__main__":
    # Example usage
    prompts = ["Please explain the concept of prompting in LLMs.",
               "Please explain the concept of prompting in LLMs in simple terms. Use a simple example.",
               "Please explain the concept of prompting in LLMs in advanced terms. Use an advanced example."]
    with open("atg/base_models/model_output.txt", "w") as f:
        for prompt in prompts:
            response = chat_with_model(prompt, 200)
            f.write(f"PROMPT:\n{prompt}\n\n")
            f.write(f"RESPONSE:\n{response}\n\n________________________________\n")
    
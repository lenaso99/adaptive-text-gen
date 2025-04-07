import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import openai
import re
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from transformers import pipeline

difficulty = "intermediate" # initial difficulty level of the model
# Options: "beginner", "intermediate", "advanced"; or more finegrained!
openai_chat = False # if True, use OpenAI chat model; if False, use Huggingface model

load_dotenv("../.env")
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
except:
    openai_chat = False # if no openai key passed, automatically do not use gpt model
hf_token = os.getenv("HUGGINGFACE_TOKEN")

login(token=hf_token)

print("Loading the model...")

if openai_chat:
    '''
    Load gpt model; gpt models are state of the art & prompting with them can be very effective
    these do not serve as a baseline necessarily as I cannot finetune them
    but they propose an upper bound for prompting
    '''
    system_prompt = f"You are a helpful assistant generating texts at an {difficulty} language level."
    chat_history = [{"role": "system", "content": system_prompt}]

    client = OpenAI(api_key=openai_api_key)

    print(">>> Welcome to the LLM chat interface! Type 'exit' or 'quit' to end the conversation.")

    while True: # start chat loop
        user_input = input(">>> You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        # Handle feedback
        if user_input.lower() in ["too simple", "too difficult", "just right"]:
            feedback = user_input.lower()
            feedback_prompt = {
                "role": "user",
                "content": f"The previous output was '{feedback}'. Please regenerate the response with { 
                    'more advanced' if feedback == 'too simple' else 
                    'simpler' if feedback == 'too difficult' else 
                    'the same' } language."
            }
            chat_history.append(feedback_prompt)
        else:
            # Regular prompt; the chat history of gpt models ensures that future responses are generated
            # with the same language as the previous ones.
            # The model will try to match the language of the previous messages in the chat history.
            chat_history.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_history
        )

        reply = response.choices[0].message.content.strip()

        print("\n>>> MODEL:\n", reply)
        
        chat_history.append({"role": "assistant", "content": reply})

        print("\n\n>>> Type 'exit' or 'quit' to end the conversation.")
        print(">>> To give the model feedback and adjust its language, rate the previous response by typing 'too simple', 'too difficult', or 'just right'.\n")


else:
    # Load model of choice
    ## LLama 2 variants
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    #model_id = "meta-llama/Llama-2-13b-chat-hf"

    ## for testing purposes:
    #model_id = "tiiuae/falcon-rw-1b"
    pipe = pipeline("text-generation", model=model_id, trust_remote_code=True) # load with simple pipeline for ease

    last_response = "" # here I store just the last response of the model and later explicitly instruct the model to use similar language to this one
    print(">>> Welcome to the LLM chat interface! Type 'exit' or 'quit' to end the conversation.")
    while True:
        print(">>> YOU:")
        user_input = input("")

        if user_input.lower() in ["exit", "quit"]:
            break

        # handle feedback
        elif user_input.lower() in ["too simple", "too difficult", "just right"]:
            feedback = user_input.lower()
            prompt = f"<INST>Previously, this answer was generated:\n\n{last_response}\n\nThis answer was rated as '{feedback}'. Rewrite the response with {'simpler' if feedback == 'too difficult' else 'more advanced' if feedback == 'too simple' else 'similar'} language.</INST>\n\n"
        else:
            # Regular prompt; the model will try to match the language of the previous response.
            # For the initial prompt, we don't have a previous response, so we just use the instruction to write in the desired language.
            feedback = None
            if last_response == "":
                prompt = f"<INST>[Instruction: Write in {difficulty} English.]\n\n{user_input}</INST>\n\n"
            else:
                prompt = f"<INST>[Instruction: Write in a similar language and difficulty as this text:\n\n{last_response}\n\n]\n\n{user_input}</INST>\n\n"

        result = pipe(prompt, max_new_tokens=300)[0]['generated_text']

        '''
        def extract_generated_text(text):
            matches = re.findall(r"</INST>\s*\n*(.*)", text, re.DOTALL)
            if matches:
                return matches[-1].strip()
            else:
                return text.strip()
        '''
            
        generated = result.split("</INST>")[-1].strip() # filter out instruction and previous response

        print("\n>>> MODEL:\n", generated)

        print("\n\n>>> Type 'exit' or 'quit' to end the conversation.")
        print(">>> To give the model feedback and adjust its language, rate the previous response by typing 'too simple', 'too difficult', or 'just right'.\n")
        last_response = result
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import openai
import re
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from transformers import pipeline

def chat_with_openai(model, key):
    '''
    Load gpt model; gpt models are state of the art & prompting with them can be very effective
    these do not serve as a baseline necessarily as I cannot finetune them
    but they propose an upper bound for prompting
    '''
    system_prompt = f"You are a helpful assistant generating texts at an {difficulty} language level."
    chat_history = [{"role": "system", "content": system_prompt}]
    user_texts = []

    client = OpenAI(api_key=openai_api_key)

    print(">>> Welcome to the LLM chat interface! Type your prompt for the model.")
    print(">>> Type 'exit' or 'quit' to end the conversation.")

    while True: # start chat loop
        user_input = input("\n>>> You: ")

        if user_input.lower() in ["exit", "quit"]:
            break
        
        try:
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

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=chat_history
                )

                reply = response.choices[0].message.content.strip()
                print("\n>>> MODEL:\n", reply)
                chat_history.append({"role": "assistant", "content": reply})
                print("\n\n>>> Type your prompt for the model.")
                print(">>> Type 'exit' or 'quit' to end the conversation.")
                print(">>> To give the model feedback and adjust its language, rate the previous response by typing 'too simple', 'too difficult', or 'just right'.")
                print(">>> To pass the model a new text, allowing it to adjust its language to you, type 'update'.")
            
            # Handle user writing for style updates 
            elif user_input.lower() in ["update"]:
                print(">>> Paste a short text you have written yourself, in your own language:")

                user_text = input(">>> Your text: ")
                user_texts.append(user_text)

                adapt_prompt = {
                    "role": "user",
                    "content": f"Please adapt the previous response to match the style of the following text: {user_text}."
                }

                chat_history.append(adapt_prompt)

                print("\n>>> Thank you! The model will now adapt its language to your writing style.")
                print("\n\n>>> Type your prompt for the model.")
                print(">>> Type 'exit' or 'quit' to end the conversation.")
                print(">>> To pass the model a new text, allowing it to adjust its language to you, type 'update'.")

            # Regular prompts...
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
                print("\n\n>>> Type your prompt for the model.")
                print(">>> Type 'exit' or 'quit' to end the conversation.")
                print(">>> To give the model feedback and adjust its language, rate the previous response by typing 'too simple', 'too difficult', or 'just right'.")
                print(">>> To pass the model a new text, allowing it to adjust its language to you, type 'update'.")
        except:
            print("Invalid API key or insufficient funds. Please check your OpenAI account and perhaps update the .env file with a valid OPENAI API key.")
            break

def chat_with_hf(pipe, token):
    '''
    Chat with 
    '''
    last_response = "" # here I store just the last response of the model and later explicitly instruct the model to use similar language to this one
    user_texts = ""
    max_sample_chars = 500
    print(">>> Welcome to the LLM chat interface! Type your prompt for the model.")
    print(">>> Type 'exit' or 'quit' to end the conversation.")

    while True: # start chat loop
        user_input = input("\n>>> You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        if user_input.lower() in ["too simple", "too difficult", "just right"]:
            feedback = user_input.lower()
            prompt = f"<INST>Previously, this answer was generated:\n\n{last_response}\n\nThis answer was rated as '{feedback}'. Rewrite the response with {'simpler' if feedback == 'too difficult' else 'more advanced' if feedback == 'too simple' else 'similar'} language.</INST>\n\n"

            result = pipe(prompt, max_new_tokens=300)[0]['generated_text']
            
            generated = result.split("</INST>")[-1].strip() # filter out instruction and previous response

            print("\n>>> MODEL:\n", generated)

            print("\n\n>>> Type your prompt for the model.")
            print(">>> Type 'exit' or 'quit' to end the conversation.")
            print(">>> To give the model feedback and adjust its language, rate the previous response by typing 'too simple', 'too difficult', or 'just right'.")
            print(">>> To pass the model a new text, allowing it to adjust its language to you, type 'update'.")
            last_response = result

        elif user_input.lower() in ["update"]:
            print(">>> Paste a short text you have written yourself, in your own language:")

            user_text = input(">>> Your text: ")

            user_texts += user_text + "\n"
            if len(user_texts) > max_sample_chars:
                user_texts = user_texts[:max_sample_chars]

            print("\n>>> Thank you! The model will now adapt its language to your writing style.")
            print("\n\n>>> Type your prompt for the model.")
            print(">>> Type 'exit' or 'quit' to end the conversation.")
            print(">>> To pass the model a new text, allowing it to adjust its language to you, type 'update'.")

        else:
            # Regular prompt; the model will try to match the language of the previous response.
            # For the initial prompt, we don't have a previous response, so we just use the instruction to write in the desired language.
            #feedback = None
            if last_response == "":
                adapt = f"[Instruction: Write in {difficulty} English.]\n\n"
            else:
                if len(user_texts) == 0:
                    adapt = f"[Instruction: Write in a similar language and difficulty as this text:\n\n{last_response}\n]"
                else:
                    adapt = f"[Instruction: Write in a similar language and difficulty as this text:\n\n{last_response}\n\nAdapt to the following user-written text:\n\n{user_texts}\n]"

            prompt = f"<INST>{adapt}\n\nUser prompt: {user_input}</INST>\n\n"
            result = pipe(prompt, max_new_tokens=300)[0]['generated_text']
            
            generated = result.split("</INST>")[-1].strip() # filter out instruction and previous response

            print("\n>>> MODEL:\n", generated)

            print("\n\n>>> Type your prompt for the model.")
            print(">>> Type 'exit' or 'quit' to end the conversation.")
            print(">>> To give the model feedback and adjust its language, rate the previous response by typing 'too simple', 'too difficult', or 'just right'.")
            print(">>> To pass the model a new text, allowing it to adjust its language to you, type 'update'.")
            last_response = result
    

if __name__ == "__main__":
    difficulty = "intermediate" # initial difficulty level of the model
    # Options: "beginner", "intermediate", "advanced"; or more finegrained!


    load_dotenv("../.env")
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_chat = True # if True, use OpenAI chat model; if False, use Huggingface model
    except:
        openai_chat = False # if no openai key passed, automatically do not use gpt model
    try:
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        hf_chat = True
    except:
        hf_chat = False

    #login(token=hf_token)

    while True:
        # handle model loading:
        print("Specify the model to load. Choose between \"Llama2-7b\", \"Llama2-13b\", \"Falcon\" and \"gpt-4o-mini\".")
        user_input = input(">>> Model to load: ")

        if user_input.lower() == "llama2-7b":
            if hf_chat:
                model = "meta-llama/Llama-2-7b-chat-hf"
                platform = "Huggingface"
            else:
                print("Model is gated and requires access, but Huggingface token not found. Please set the HUGGINGFACE_TOKEN environment variable in .env file or load a different model.")
                continue
        elif user_input.lower() == "llama2-13b":
            if hf_chat:
                model = "meta-llama/Llama-2-13b-chat-hf"
                platform = "Huggingface"
            else:
                print("Model is gated and requires access, but Huggingface token not found. Please set the HUGGINGFACE_TOKEN environment variable in .env file or load a different model.")
                continue
        elif user_input.lower() == "falcon":
            model = "tiiuae/falcon-7b-instruct"
            platform = "Huggingface"
        elif user_input.lower() == "gpt-4o-mini":
            if openai_chat:
                model = "gpt-4o-mini"
                platform = "OpenAI"
            else:
                print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in .env file or load a different model.")
                continue
        else:
            print("Invalid model name. Please try again.")
            continue
               

        print(f"Load model {model}? Enter \"yes\" to load the model or \"no\" to choose another model.")
        user_input = input(">>> Load model? ")
        if user_input.lower() == "yes":
            break
        elif user_input.lower() == "no":
            continue
        else:
            print("Invalid input. Please try again.")
            continue
    
    print(f"Loading model {model}...")
    if platform == "Huggingface":
        if "llama" in model:
            try:
                pipe = pipeline("text-generation", model=model, trust_remote_code=True)
                chat_with_hf(pipe, hf_token)
            except:
                print(f"Model is gated and requires access. Please request access to {model} on huggingface.co.")
    else:
        chat_with_openai(model, openai_api_key)

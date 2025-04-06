import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import openai
from openai import OpenAI
from transformers import pipeline

difficulty = "intermediate"
openai_chat = True

openai_api_key = "..."  # Replace with your OpenAI API key
hf_token = "..."  # Replace with your Hugging Face token

print("Loading the model...")

if openai_chat:
    system_prompt = f"You are a helpful assistant generating texts at an {difficulty} language level."
    chat_history = [{"role": "system", "content": system_prompt}]

    client = OpenAI(
        api_key=openai_api_key
    )

    print(">>> Welcome to the LLM chat interface! Type 'exit' or 'quit' to end the conversation.")

    while True:
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
            # Regular prompt
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
    #model_id = "meta-llama/Llama-2-7b-chat-hf"
    #model_id = "meta-llama/Llama-2-13b-chat-hf"

    ## for testing purposes:
    model_id = "tiiuae/falcon-rw-1b"
    pipe = pipeline("text-generation", model=model_id, trust_remote_code=True)

    last_response = ""
    print("Welcome to the LLM chat interface! Type 'exit' or 'quit' to end the conversation.")
    while True:
        user_input = input(">>> YOU: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        elif user_input.lower() in ["too simple", "too difficult", "just right"]:
            feedback = user_input.lower()
            prompt = f"The previous answer was rated as '{feedback}'. Rewrite the response with {'simpler' if feedback == 'too difficult' else 'more advanced' if feedback == 'too simple' else 'similar'} language:\n\n{last_response}"
        else:
            feedback = None
            prompt = f"[Instruction: Write in {difficulty} English.]\n\n{user_input}"

        result = pipe(prompt, max_new_tokens=300)[0]['generated_text']
        print("\n>>> MODEL:\n", result)

        print("\n\n>>> Type 'exit' or 'quit' to end the conversation.")
        print(">>> To give the model feedback and adjust its language, rate the previous response by typing 'too simple', 'too difficult', or 'just right'.\n")
        last_response = result
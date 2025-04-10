import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import datetime
import openai
import pandas as pd
import re
import textstat
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from transformers import pipeline

def get_gpt_ranking(prompt, command, text1, text2, key):
    client = OpenAI(api_key=key)
    gpt_prompt = f"""You will now be given 2 texts that were written as a response to the same prompt.
                    Your task is rank these two texts on difficulty.\n
                    PROMPT:
                    {prompt}

                    TEXT 1:
                    {text1}

                    TEXT 2:
                    {text2}

                    Please determine whether TEXT 2 is more advanced, simpler or very similar to TEXT 1. Provide a short explanation for your ranking.
                    Format your response following this example format, ensuring you only use "simpler", "more advanced" or "similar" as the ranking:

                    TEXT 2: <simpler/more advanced/similar> 
                    EXPLANATION: <explanation>
                    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful language teacher assistent that helps evaluating texts based on their complexity."},
            {"role": "user", "content": gpt_prompt},
        ]
    )
    reply = response.choices[0].message.content.strip()
    # parsing the response:
    # Extract the ranking and explanation from the response
    ranking_match = re.search(r"TEXT 2:\s*(simpler|more advanced|similar)", reply)
    explanation_match = re.search(r"EXPLANATION:\s*(.+)", reply)

    if ranking_match and explanation_match:
        ranking = ranking_match.group(1)
        explanation = explanation_match.group(1)
    else:
        ranking = None
        explanation = None

    return {
        "prompt": prompt,
        "command": command,
        "response": text2,
        "fkgl": textstat.flesch_kincaid_grade(text2),
        "fre": textstat.flesch_reading_ease(text2),
        "ranking": ranking,
        "explanation": explanation
    }

def get_updating_eval(prompt, text1, text2, update, key):
    client = OpenAI(api_key=key)
    gpt_prompt = f"""You will now be given 2 texts that were written as a response to the same prompt.
                    Additionally, you will be given a user-written text that the model should adapt to.
                    Your task is to determine which of the 2 texts is more similar to the user-written text.\n
                    PROMPT:
                    {prompt}

                    USER_WRITTEN TEXT:
                    {update}


                    TEXT 1:
                    {text1}

                    TEXT 2:
                    {text2}

                    Please determine whether TEXT 2 is more similar, less similar, or equally similar to the USER_WRITTEN TEXT compared to TEXT 1. Provide a short explanation for your decision.
                    Format your response following this example format, ensuring you only use "more similar", "less similar" or "equally similar" as the ranking:

                    TEXT 2: <more similar/less similar/equally similar> 
                    EXPLANATION: <explanation>
                    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful language teacher assistent that helps evaluating texts based on their complexity."},
            {"role": "user", "content": gpt_prompt},
        ]
    )
    reply = response.choices[0].message.content.strip()
    # parsing the response:
    # Extract the ranking and explanation from the response
    eval_match = re.search(r"TEXT 2:\s*(more similar|less similar|equally similar)", reply)
    explanation_match = re.search(r"EXPLANATION:\s*(.+)", reply)

    if eval_match and explanation_match:
        ranking = eval_match.group(1)
        explanation = explanation_match.group(1)
    else:
        ranking = None
        explanation = None

    return {
        "prompt": prompt,
        "base_response": text1,
        "fkgl_base": textstat.flesch_kincaid_grade(text1),
        "fre_base": textstat.flesch_reading_ease(text1),
        "update_text": update,
        "fkgl_text": textstat.flesch_kincaid_grade(update),
        "fre_text": textstat.flesch_reading_ease(update),
        "updated_response": text2,
        "fkgl_update": textstat.flesch_kincaid_grade(text2),
        "fre_update": textstat.flesch_reading_ease(text2),
        "evaluation": ranking,
        "explanation": explanation
    }


def chat_with_openai(model, key, evaluation_mode):
    '''
    A function to chat with OpenAI's chat models.

    As parameters, it takes:
        - model [str]: the gpt model to be used
        - key [str]: the openai api key to be used for the model
        - evaluation_mode [bool]: if True, the model will be evaluated

    If evaluation_mode is False, returns nothing, but executes the chat with the model
    If evaluation_mode is True, returns 
        - a dataframe with the (gpt) rankings and scores of the responses
        - a dataframe with the evaluation of updating the responses with user text, plus their scores
    '''
    system_prompt = f"You are a helpful assistant generating texts at an {difficulty} language level."
    chat_history = [{"role": "system", "content": system_prompt}]
    user_texts = []

    client = OpenAI(api_key=key)

    if evaluation_mode:
        last_response_stored = ""
        last_prompt_stored = ""
        ranking_df = pd.DataFrame(columns=["prompt", "command", "response", "fkgl", "fre" "ranking", "explanation"])
        updating_eval_df = pd.DataFrame(columns=["prompt", "base_response", "fkgl_base", "fre_base", "update_text", "fkgl_text", "fre_text", "updated_response", "fkgl_update", "fre_update", "evaluation", "explanation"])

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
                    model=model,
                    messages=chat_history
                )

                reply = response.choices[0].message.content.strip()

                # if the evaluation mode is turned on, do evaluation
                if evaluation_mode:
                    # ranking: compare new response to last one
                    command = "simpler" if feedback == "too difficult" else "more advanced" if feedback == "too simple" else "similar"
                    gpt_ranking = get_gpt_ranking(last_prompt_stored, command, last_response_stored, reply, key)
                    gpt_ranking_df = pd.DataFrame([gpt_ranking])
                    ranking_df = pd.concat([ranking_df, gpt_ranking_df], ignore_index=True)

                    last_response_stored = reply # store the last response for evaluation

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

                if evaluation_mode:
                    chat_history_2 = chat_history.copy()
                    chat_history_2.append({"role": "user", "content": last_prompt_stored})
                    response = client.chat.completions.create(
                        model=model,
                        messages=chat_history_2
                    )

                    reply = response.choices[0].message.content.strip()

                    updating_eval = get_updating_eval(last_prompt_stored, last_response_stored, reply, user_text, key)
                    updating_eval_df = pd.concat([updating_eval_df, pd.DataFrame([updating_eval])], ignore_index=True)
                    
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
                    model=model,
                    messages=chat_history
                )


                reply = response.choices[0].message.content.strip()

                # if the evaluation mode is turned on, do evaluation
                if evaluation_mode:
                    
                    last_prompt_stored = user_input

                    if len(last_response_stored) == 0: # if this is the first prompt...
                        # simply store it as initial prompt in the ranking dataframe
                        first_row = pd.DataFrame([{"prompt": user_input, "command": "initial", "response": reply, "ranking": None, "explanation": None}])
                        ranking_df = pd.concat([ranking_df, first_row], ignore_index=True)

                    last_response_stored = reply # store the last response for evaluation

                print("\n>>> MODEL:\n", reply)
                chat_history.append({"role": "assistant", "content": reply})
                print("\n\n>>> Type your prompt for the model.")
                print(">>> Type 'exit' or 'quit' to end the conversation.")
                print(">>> To give the model feedback and adjust its language, rate the previous response by typing 'too simple', 'too difficult', or 'just right'.")
                print(">>> To pass the model a new text, allowing it to adjust its language to you, type 'update'.")
        except:
            print("Invalid API key or insufficient funds. Please check your OpenAI account and perhaps update the .env file with a valid OPENAI API key.")
            break

    if evaluation_mode:
        return ranking_df, updating_eval_df

def chat_with_hf(model, token, evaluation_mode, key=None):
    ''''
    A function to chat with any model from huggingface.

    As parameters, it takes:
        - model [str]: the model to be used
        - token [str]: the huggingface token to be used for model loading
        - evaluation_mode [bool]: if True, the model will be evaluated

    If evaluation_mode is False, returns nothing, but executes the chat with the model
    If evaluation_mode is True, returns 
        - a dataframe with the (gpt) rankings and scores of the responses
        - a dataframe with the evaluation of updating the responses with user text, plus their scores
    '''
    pipe = pipeline("text-generation", model=model, trust_remote_code=True, token=token)

    last_response = "" # here I store just the last response of the model and later explicitly instruct the model to use similar language to this one
    user_texts = ""
    max_sample_chars = 500

    if evaluation_mode:
        last_response_stored = ""
        last_prompt_stored = ""
        ranking_df = pd.DataFrame(columns=["prompt", "command", "response", "ranking", "explanation"])
        updating_eval_df = pd.DataFrame(columns=["prompt", "base_response", "update_text", "updated_response", "evaluation", "explanation"])

    print(">>> Welcome to the LLM chat interface! Type your prompt for the model.")
    print(">>> Type 'exit' or 'quit' to end the conversation.")

    while True: # start chat loop
        user_input = input("\n>>> You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        if user_input.lower() in ["too simple", "too difficult", "just right"]:
            # Handle feedback
            feedback = user_input.lower()
            prompt = f"<INST>Previously, this answer was generated:\n\n{last_response}\n\nThis answer was rated as '{feedback}'. Rewrite the response with {'simpler' if feedback == 'too difficult' else 'more advanced' if feedback == 'too simple' else 'similar'} language.</INST>\n\n"

            result = pipe(prompt, max_new_tokens=300)[0]['generated_text']
            
            generated = result.split("</INST>")[-1].strip() # filter out instruction and previous response

            if evaluation_mode:
                # ranking: compare new response to last one
                command = "simpler" if feedback == "too difficult" else "more advanced" if feedback == "too simple" else "similar"
                gpt_ranking = get_gpt_ranking(last_prompt_stored, command, last_response_stored, generated, key)
                gpt_ranking_df = pd.DataFrame([gpt_ranking])
                ranking_df = pd.concat([ranking_df, gpt_ranking_df], ignore_index=True)

                last_response_stored = generated # store the last response for evaluation

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

            if evaluation_mode:
                # generate new response
                adapt = f"[Instruction: Write in a similar language and difficulty as this text:\n\n{last_response}\n\nAdapt to the following user-written text:\n\n{user_texts}\n]"

                prompt = f"<INST>{adapt}\n\nUser prompt: {last_prompt_stored}</INST>\n\n"
                result = pipe(prompt, max_new_tokens=300)[0]['generated_text']

                # compare to old response
                updating_eval = get_updating_eval(last_prompt_stored, last_response_stored, result, user_text, key)
                updating_eval_df = pd.concat([updating_eval_df, pd.DataFrame([updating_eval])], ignore_index=True)

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

            if evaluation_mode:
                last_prompt_stored = user_input

                if len(last_response_stored) == 0: # if this is the first prompt...
                    # simply store it as initial prompt in the ranking dataframe
                    first_row = pd.DataFrame([{"prompt": user_input, "command": "initial", "response": generated, "ranking": None, "explanation": None}])
                    ranking_df = pd.concat([ranking_df, first_row], ignore_index=True)
                
                last_response_stored = generated # store the last response for evaluation

            print("\n>>> MODEL:\n", generated)

            print("\n\n>>> Type your prompt for the model.")
            print(">>> Type 'exit' or 'quit' to end the conversation.")
            print(">>> To give the model feedback and adjust its language, rate the previous response by typing 'too simple', 'too difficult', or 'just right'.")
            print(">>> To pass the model a new text, allowing it to adjust its language to you, type 'update'.")
            last_response = result
    
    if evaluation_mode:
        return ranking_df, updating_eval_df

if __name__ == "__main__":
    #TODO: add a sequence of commands to the FKGL calculation so the scores actually mean something;
    #TODO: just add score calculation to the other two steps because they don't make sense in isolation
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

    print("Do you want to turn on evaluation mode? (yes/no)")
    user_input = input(">>> ")
    if user_input.lower() == "yes":
        if openai_chat:
            evaluation_mode = True
        else:
            print("Evaluation mode is only available if OpenAI API key is set. Please set the OPENAI_API_KEY environment variable in .env file. Evaluation mode disabled.")
            evaluation_mode = False
    else:
        evaluation_mode = False

    #login(token=hf_token)

    while True:
        # handle model loading:
        print("Specify the model to load. Choose between \"Llama2-7b\", \"Llama2-13b\", \"Falcon\" and \"gpt-4o-mini\".")
        user_input = input(">>> Model to load: ")

        if user_input.lower() == "llama2-7b" or user_input.lower() == "llama2":
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
        elif user_input.lower() == "falcon" or user_input.lower() == "falcon-7b":
            model = "tiiuae/falcon-7b-instruct"
            platform = "Huggingface"
        elif user_input.lower() == "gpt-4o-mini" or user_input.lower() == "gpt-4o" or user_input.lower() == "gpt":
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
        if evaluation_mode:
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            ranking, updating = chat_with_hf(model, hf_token, evaluation_mode=True, key=openai_api_key)

            ranking.to_excel(f"results/{model.split("/")[1]}/ranking_scores_{current_datetime}.xlsx", index=False)
            updating.to_excel(f"results/{model.split("/")[1]}/updating_scores_{current_datetime}.xlsx", index=False)
        else:
            chat_with_hf(model, hf_token, evaluation_mode=False)

    else:
        if evaluation_mode:
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            ranking, updating = chat_with_openai(model, openai_api_key, evaluation_mode=True)

            ranking.to_excel(f"results/{model}/ranking_scores_{current_datetime}.xlsx", index=False)
            updating.to_excel(f"results/{model}/updating_scores_{current_datetime}.xlsx", index=False)
        else:
            chat_with_openai(model, openai_api_key, evaluation_mode=False)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import datetime
import openai
import pandas as pd
import re
import textstat
from dotenv import load_dotenv
from openai import OpenAI

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


def chat_with_openai(model, key, prompt_mode = "zero-shot", evaluation_mode=False):
    '''
    A function to chat with OpenAI's chat models.

    As parameters, it takes:
        - model [str]: the gpt model to be used
        - prompt_mode [str]: the prompt mode to be used ("zero-shot", "few-shot" or "reasoning")
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
        ranking_df = pd.DataFrame(columns=["prompt", "command", "response", "fkgl", "fre", "ranking", "explanation"])
        updating_eval_df = pd.DataFrame(columns=["prompt", "base_response", "fkgl_base", "fre_base", "update_text", "fkgl_text", "fre_text", "updated_response", "fkgl_update", "fre_update", "evaluation", "explanation"])

    print(">>> Welcome to the LLM chat interface! Type your prompt for the model.")
    print(">>> Type 'exit' or 'quit' to end the conversation.")

    while True: # start chat loop
        
        user_input = input("\n>>> You: ")

        if user_input.lower() in ["exit", "quit"]:
            break
        elif user_input.lower() in ["reset"]:
            print(">>> Resetting chat history...")
            chat_history = [{"role": "system", "content": system_prompt}]
            user_texts = []
            print("\n>>> Type your prompt for the model.")
            continue

        try:
        # Handle feedback
            if user_input.lower() in ["too simple", "too easy", "too difficult", "too advanced", "just right"]:
                feedback = user_input.lower()

                if prompt_mode == "zero-shot":
                    feedback_prompt = {
                        "role": "user",
                        "content": f"The previous output was '{feedback}'. Please regenerate the response with { 
                            'more advanced' if feedback == 'too simple' or feedback == "too easy" else 
                            'simpler' if feedback == 'too difficult' or feedback == "too advanced" else 
                            'the same' } language."
                    }

                elif prompt_mode == "few-shot":
                    feedback_prompt = {
                        "role": "user",
                        "content": f"""The previous output was '{feedback}'. Please regenerate the response with { 
                            'more advanced' if feedback == 'too simple' else 
                            'simpler' if feedback == 'too difficult' else 
                            'the same' } language. Here are some examples of how to do this:

                            Example 1:
                            Response: The process of photosynthesis allows plants to convert sunlight into energy.
                            Feedback: too difficult
                            Adapted: Photosynthesis is how plants turn sunlight into energy.

                            Example 2:
                            Response: The capital of France is Paris, which is known for its art, culture, and history.
                            Feedback: too simple
                            Adapted: Paris, the capital of France, is renowned not only for its pivotal role in shaping European intellectual and artistic movements but also for its rich cultural heritage, which spans centuries of historical significance in fields ranging from philosophy and literature to visual arts and architecture

                            Example 3:
                            Response: In the context of LLMs, fine-tuning refers to adjusting the parameters of a model based on a smaller, task-specific dataset.
                            Feedback: just right
                            Adapted: Fine-tuning is the process of refining a pre-trained model’s parameters by exposing it to additional data focused on a specific task.

                            Now, it is your turn:
                            """
                    }

                elif prompt_mode == "reasoning":
                    feedback_prompt = {
                        "role": "user",
                        "content": f"""The previous output was '{feedback}'. Please regenerate the response with { 
                            'more advanced' if feedback == 'too simple' else 
                            'simpler' if feedback == 'too difficult' else 
                            'the same' } language.
                            Let's think about this step by step!"""
                    }
                else:
                    raise ValueError("Invalid prompt mode. Choose between 'zero-shot', 'few-shot' or 'reasoning'.")
                
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
                print(">>> Type 'reset' to reset the chat history.")
                print(">>> To give the model feedback and adjust its language, rate the previous response by typing 'too simple', 'too difficult', or 'just right'.")
                print(">>> To pass the model a new text, allowing it to adjust its language to you, type 'update'.")
            
            # Handle user writing for style updates 
            elif user_input.lower() in ["update"]:
                print(">>> Paste a short text you have written yourself, in your own language:")

                user_text = input(">>> Your text: ")
                user_texts.append(user_text)

                print("\n>>> Thank you! The model will now adapt its language to your writing style...")

                if prompt_mode == "zero-shot":
                    adapt_prompt = {
                        "role": "user",
                        "content": f"Please adapt the previous response to match the style of the following text: {user_text}."
                    }

                elif prompt_mode == "few-shot":
                    adapt_prompt = {
                        "role": "user",
                        "content": f"""Please adapt the previous response to match the style of the following text: {user_text}.
                            Here are some examples of how to do this:
                            
                            Example 1:
                            Response: Photosynthesis is a process by which plants use sunlight to produce energy.
                            User text: I like how plants eat light and make their own food. It's kinda cool!
                            Adapted: Plants are cool because they can eat light and turn it into food — it's called photosynthesis.

                            Example 2:
                            Response: The capital of France is Paris, which is known for its art, culture, and history.
                            User text: I’ve always wanted to see the Eiffel Tower. Paris must be full of amazing stories.
                            Adapted: Paris is full of stories and beautiful places — it’s the capital of France and home to the Eiffel Tower.

                            Example 3:
                            Response: In the context of LLMs, fine-tuning refers to adjusting the parameters of a model based on a smaller, task-specific dataset.
                            User text: So fine-tuning is like making the model better at what it does by training it on more relevant data, right?
                            Adapted: Fine-tuning is the process of refining a pre-trained model’s parameters by exposing it to additional data focused on a specific task.
                            """
                    }
                    
                elif prompt_mode == "reasoning":
                    adapt_prompt = {
                        "role": "user",
                        "content": f"""Please adapt the previous response to match the style of the following text: {user_text}.
                            Let's think about this step by step!"""
                    }
                else:
                    raise ValueError("Invalid prompt mode. Choose between 'zero-shot', 'few-shot' or 'reasoning'.")

                chat_history.append(adapt_prompt) 

                chat_history.append({"role": "user", "content": last_prompt_stored})
                response = client.chat.completions.create(
                    model=model,
                    messages=chat_history
                )

                reply = response.choices[0].message.content.strip()
                if evaluation_mode:
                    updating_eval = get_updating_eval(last_prompt_stored, last_response_stored, reply, user_text, key)
                    updating_eval_df = pd.concat([updating_eval_df, pd.DataFrame([updating_eval])], ignore_index=True)

                last_response_stored = reply # store the last response for evaluation
                print("\n>>> MODEL:\n", reply)
                print("\n\n>>> Type your prompt for the model.")
                print(">>> Type 'exit' or 'quit' to end the conversation.")
                print(">>> Type 'reset' to reset the chat history.")
                print(">>> To give the model feedback and adjust its language, rate the previous response by typing 'too simple', 'too difficult', or 'just right'.")
                print(">>> To pass the model a new text, allowing it to adjust its language to you, type 'update'.")

            # Regular prompts...
            else:
                # Regular prompt; the chat history of gpt models ensures that future responses are generated
                # with the same language as the previous ones.
                # The model will try to match the language of the previous messages in the chat history.

                if prompt_mode == "zero-shot" or prompt_mode == "reasoning":
                    # there's no need for reasoning on a simple prompt, so we can use the same prompt for both modes
                    prompt = user_input

                elif prompt_mode == "few-shot":
                    prompt = f"""Here are some examples of how to respond to prompts:
                        
                        Example 1: 
                        Prompt: How does photosynthesis work?
                        Response: Photosynthesis is how plants make their food using sunlight, water, and carbon dioxide.

                        Example 2:
                        Prompt: What is the capital of France?
                        Response: The capital of France is Paris, which is known for its art, culture, and history.

                        Example 3:
                        Prompt: What is fine-tuning in the context of LLMs?
                        Response: Fine-tuning is the process of adjusting a pre-trained model's parameters based on a smaller, task-specific dataset.

                        Now it is your turn:
                        Prompt: {user_input}
                        Response:
                    """

                else:
                    raise ValueError("Invalid prompt mode. Choose between 'zero-shot', 'few-shot' or 'reasoning'.")


                chat_history.append({"role": "user", "content": prompt})

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
                        first_row = pd.DataFrame([{"prompt": user_input, "command": "initial", "response": reply, "fkgl": textstat.flesch_kincaid_grade(reply), "fre": textstat.flesch_reading_ease(reply), "ranking": None, "explanation": None}])
                        ranking_df = pd.concat([ranking_df, first_row], ignore_index=True)

                    last_response_stored = reply # store the last response for evaluation

                print("\n>>> MODEL:\n", reply)
                chat_history.append({"role": "assistant", "content": reply})
                print("\n\n>>> Type your prompt for the model.")
                print(">>> Type 'exit' or 'quit' to end the conversation.")
                print(">>> Type 'reset' to reset the chat history.")
                print(">>> To give the model feedback and adjust its language, rate the previous response by typing 'too simple', 'too difficult', or 'just right'.")
                print(">>> To pass the model a new text, allowing it to adjust its language to you, type 'update'.")
        except:
            print("Invalid API key or insufficient funds. Please check your OpenAI account and perhaps update the .env file with a valid OPENAI API key.")
            break

    if evaluation_mode:
        return ranking_df, updating_eval_df
    
if __name__ == "__main__":
    difficulty = "intermediate" # initial difficulty level of the model
    # Options: "beginner", "intermediate", "advanced"; or more finegrained!

    load_dotenv("../.env")
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_chat = True # if True, use OpenAI chat model; if False, use Huggingface model
    except:
        openai_chat = False # if no openai key passed, automatically do not use gpt model
    
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

    model = "gpt-4o-mini"

    while True:
        print(f"Load model {model}. Enter \"proceed\" to proceed. Enter \"exit\" to abort.")
        user_input = input(">>> Load model? ")
        if user_input.lower() == "proceed":
            break
        elif user_input.lower() == "exit":
            break
        else:
            print("Invalid input. Please try again.")
            continue
    
    print("Choose the prompt mode between \"zero-shot\", \"few-shot\" and \"reasoning\" (for zero-shot reasoning):")
    prompt_mode = input(">>> Prompt mode: ").lower()
    if prompt_mode not in ["zero-shot", "few-shot", "reasoning"]:
        print("Invalid input. Please try again.")
        exit()

    print(f"Loading model {model}...")

    if evaluation_mode:
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        ranking, updating = chat_with_openai(model, openai_api_key, prompt_mode=prompt_mode, evaluation_mode=True)

        if not ranking.empty:
            ranking.to_excel(f"results/{model}_{prompt_mode}_ranking_scores_{current_datetime}.xlsx", index=False)
            print(f"Ranking dataframe saved as results/{model}_{prompt_mode}_ranking_scores_{current_datetime}.xlsx")
        else:
            print("Ranking dataframe is empty. Skipping save.")

        if not updating.empty:
            updating.to_excel(f"results/{model}_{prompt_mode}_updating_scores_{current_datetime}.xlsx", index=False)
            print(f"Updating dataframe saved as results/{model}_{prompt_mode}_updating_scores_{current_datetime}.xlsx")
        else:
            print("Updating dataframe is empty. Skipping save.")
    else:
        chat_with_openai(model, openai_api_key, prompt_mode=prompt_mode, evaluation_mode=False)
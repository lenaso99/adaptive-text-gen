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
        model="gpt-4o-mini",,
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
        model="gpt-4o-mini",
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


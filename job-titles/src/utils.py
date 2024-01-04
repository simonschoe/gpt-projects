"""Utility functions"""
import ast
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tiktoken import encoding_for_model

# load api key
load_dotenv()


def collate_examples(g):
    '''Collate the examples into a dictionary.'''
    cols = g.columns[:-1]
    return {c: g[c].values for c in cols}


def create_message(**kwargs):
    '''Create a messages for the API call.'''
    model = kwargs['model']
    encoding = encoding_for_model(model)

    # create messages
    if isinstance(kwargs['title'], np.ndarray):
        kwargs['title'] = ''.join([f"\n{i+1}. {t}" for i, t in enumerate(kwargs['title'])])

    messages = [
        {"role": "system",
         "content": kwargs['p_sys'].read_text().format(**kwargs)},
        {"role": "user",
         "content": kwargs['p_user'].read_text().format(**kwargs)},
    ]

    # count token
    if model == 'gpt-3.5-turbo-0301':
        tok_per_prompt = 4      # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tok_per_name = -1       # if there's a name, the role is omitted
    elif model in ['gpt-4-0314', 'gpt-4-0613', 'gpt-4-1106-preview']:
        tok_per_prompt = 3
        tok_per_name = 1
    else:
        raise NotImplementedError(
            f"""
            count_token() is not implemented for model {model}.
            See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.
            """
        )
    n_tok = 0
    for m in messages:
        n_tok += tok_per_prompt
        for key, value in m.items():
            n_tok += len(encoding.encode(value))
            if key == "name":
                n_tok += tok_per_name
    n_tok += 3  # every reply is primed with <|start|>assistant<|message|>

    return (messages, n_tok)


def get_completion(messages, group_id, titles_ids, model, p_user):
    '''Get the completion from the API or from a preexisting file.'''
    fpath = Path('..', 'output', f'{p_user.stem}', f'{group_id}.json')
    if not fpath.parent.exists():
        fpath.parent.mkdir()
    if not fpath.exists():
        retries = 0
        success = False
        while not success and retries < 10:
            try:
                client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={ "type": "json_object" },
                    max_tokens=600,
                    temperature=0
                )
                success = True
            except Exception as e:
                print(e)
                retries += 1
        result = {
            'ids': titles_ids.tolist(),
            'output': response.choices[0].message.content,
            'finish_reason': response.choices[0].finish_reason,
            'tok_prompt': response.usage.prompt_tokens,
            'tok_completion': response.usage.completion_tokens
        }
        with fpath.open('w', encoding='utf-8') as f:
            json.dump(result, f)
        return result
    else:
        result = json.loads(fpath.read_bytes())
        return result

def extract_model_answers(response):
    '''Extract the answers from the JSON API response.'''''
    try:
        response = ast.literal_eval(response['output'])
        if len(response.keys()) == 1:
            return response['category']
        return [r['category'] for r in response.values()]
    except Exception as e:
        print(e)
        return None
    
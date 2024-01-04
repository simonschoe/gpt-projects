"""Utility functions for working with the OpenAI API."""

import ast
import json
import re
from pathlib import Path

from openai import ChatCompletion
from tiktoken import encoding_for_model


event_codes = {
    'a.': 'change in the top management team',
    'b.': 'substantial changes in the workforce',
    'c.': 'changes in production capacities',
    'd.': 'shifts in the firm\'s strategy',
    'e.': 'capital increases, share buyback programs, or repayment of debt',
    'f.': 'dividend announcements',
    'g.': 'M&A activities, divestment, or joint venture agreement',
    'h.': 'changes in security analyst forecasts or recommendations',
    'i.': 'credit rating change',
    'j.': 'legal fines, lawsuit, or regulatory changes',
    'k.': 'announcement of financial results',
    'l.': 'product recall announcements',
    'm.': 'other'
}


def create_message(**kwargs):
    """Create messages for the OpenAI API."""

    model = kwargs['model']
    thresh = kwargs['thresh']
    encoding = encoding_for_model(model)

    # shorten article
    if len(encoding.encode(kwargs['body'])) > thresh:
        trimmed = True
        kwargs['body'] = encoding.decode(encoding.encode(kwargs['body'])[:thresh]) + ' [...]'
    else:
        trimmed = False

    # create messages
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

    return (messages, trimmed, n_tok)


def get_completion(messages, news_id, model, p_user):
    """ Get completion from OpenAI API."""
    fpath = Path('..', 'output', f'{p_user.stem}', f'{news_id}.json')
    if not fpath.parent.exists():
        fpath.parent.mkdir()
    if not fpath.exists():
        retries = 0
        success = False
        while not success and retries < 10:
            try:
                response = ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=300,
                    temperature=0,
                    response_format={ "type": "json_object" }
                )
                success = True
            except Exception as e:
                print(e)
                retries += 1
        result = {
            'output': response['choices'][0]['message']['content'],
            'tok_prompt': response['usage']['prompt_tokens'],
            'tok_completion': response['usage']['completion_tokens']
        }
        with fpath.open('w', encoding='utf-8') as f:
            json.dump(result, f)
        return result
    else:
        result = json.loads(fpath.read_bytes())
        return result


def extract_model_answers(x):
    """ Extract model answers from OpenAI API response."""
    x  = x['output']
    x = re.sub(r': null', ': ""', x, re.I)
    x = ast.literal_eval(x)
    try:
        relevant = x['final_answer'].lower() == 'yes'
        codes = x['event_type']
        codes = [re.sub(r"\..*?$", '', code, re.I) for code in codes]
        codes = {k: k[0].lower() in codes for k in event_codes}
        return relevant, codes
    except Exception as e:
        print(e, x)

def extract_model_answers_s1(response):
    """ Extract model answers from OpenAI API response."""
    try:
        (e1, a1), (e2, a2) = \
            [(re.search('(?<=- Short explanation:).*', step, flags=re.I), re.search('(?<=- Answer:).*', step, flags=re.I))
            for step in response['output'].split("\nStep 2:")]
        e1 = e1[0].strip() if e1 else e1
        a1 = a1[0].strip() if a1 else a1
        a1 = re.search('yes', a1, flags=re.I)[0] if (a1 and re.search('yes', a1, flags=re.I)) else a1
        e2 = e2[0].strip() if e2 else e2
        a2 = a2[0].strip() if a2 else a2
        a2 = re.search('yes', a2, flags=re.I)[0] if (a2 and re.search('yes', a2, flags=re.I)) else a2
        return e1, a1, e2, a2
    except ValueError as e:
        print(e)
        return [None] * 4


def extract_model_answers_s2(response):
    """ Extract model answers from OpenAI API response."""
    try:
        codes = re.findall('\\b[a-m]\\b\\.', response['output'])
        codes = {code: event_codes[code] for code in codes}
        if 'm.' in codes.keys():
            codes['m.'] = re.search('(?<=m. )(other - )?(.*?)$', response['output'].replace("\n", " "), flags=re.I).groups()[-1]
        return codes
    except Exception as e:
        print(e, '\n', response['output'])
        return None

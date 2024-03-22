conversations = ['Sale: Hello, how can I help with your insurance needs today?', 'Customer: Not interested',
'Sale: Hello, how can I help with your insurance needs today?', 'Customer: I am not interested in your insurance.',
'Sale: Hello, how can I help with your insurance needs today?', 'Customer: Nah I got other stuff to do.',
'Sale: Hello, how can I help with your insurance needs today?', 'Customer: Let me see, actually no.',
'Sale: Hello, how can I help with your insurance needs today?', 'Customer: Interesting, what sort of product do you have?']

from itertools import permutations
# Generate all unique combination pairs of conversations
conversation_pairs = list(permutations(conversations, 2))
annos = list(permutations(range(len(conversations)), 2))

synonyms = ['mode rude', 'ruder', 'more impolite', 'more discourteous', 'more ill-mannered', 'more disrespectful', 'more insolent', 'more impudent','more uncivil','more impertinent','more vulgar','more crass','more boorish','more churlish','more abrasive', 'more tactless', 'more offensive', 'more insulting', 'more derogatory', 'more disparaging', 'more contemptuous']

query_template = """Compare customers' response in the two conversations:
Conversation A: {conversation_a}
Conversation B: {conversation_b}
Is customer A {desc} than customer B?
{answer}"""

reverse_query_template = """Compare customers' response in the two conversations:
Conversation B: {conversation_b}
Conversation A: {conversation_a}
Is customer A {desc} than customer B?
{answer}"""

split_query = lambda query: (('\n').join(query.split('\n')[:-1]), query.split('\n')[-1])

import random

def map_to_answer(anno):
    if anno[0] < anno[1]:
        return 'Yes'
    else:
        return 'No'


def prepare_sft_dataset(conversation_pairs, annotations, query_template, reverse_query_template, synonyms):
    dataset = []

    for (conversation_a, conversation_b), annotation in zip(conversation_pairs, annotations):

        for syn in synonyms:
            query = query_template.format(
            conversation_a=conversation_a,
            conversation_b=conversation_b,
            answer=map_to_answer(annotation),
            desc = random.choice(synonyms)
            )
            query, answer = split_query(query)
            dataset.append({'prompt': query, 'completion': answer})

            reverse_query = reverse_query_template.format(
            conversation_a = conversation_a,
            conversation_b = conversation_b,
            answer = map_to_answer(annotation),
            desc = random.choice(synonyms)
            )
            query, answer = split_query(query)
            dataset.append({'prompt': query, 'completion': answer})

    return dataset

sft_dataset = prepare_sft_dataset(conversation_pairs, annos, query_template, reverse_query_template, synonyms)


import pandas as pd
from datasets import Dataset
from huggingface_hub import login
login("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")
df = pd.DataFrame(sft_dataset, columns=['prompt','completion'])
dataset = Dataset.from_pandas(df)
dataset.push_to_hub("Ksgk-fy/alignment-test-sft")
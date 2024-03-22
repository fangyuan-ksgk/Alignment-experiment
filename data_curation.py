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

{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "How far is the Moon from Earth?"}, {"role": "assistant", "content": "..."}]}


def prepare_sft_dataset(conversation_pairs, annotations, query_template, reverse_query_template, synonyms, mode = 1):
    dataset = []

    for (conversation_a, conversation_b), annotation in zip(conversation_pairs, annotations):

        for syn in synonyms:
            query = query_template.format(
            conversation_a=conversation_a,
            conversation_b=conversation_b,
            answer=map_to_answer(annotation),
            desc = syn
            )
            query, answer = split_query(query)
            if mode == 1:
                mes = {"prompt": query, "completion": answer}
            else:
                mes = {"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": query}, {"role": "assistant", "content": answer}]}
            dataset.append(mes)

            reverse_query = reverse_query_template.format(
            conversation_a = conversation_a,
            conversation_b = conversation_b,
            answer = map_to_answer(annotation),
            desc = syn
            )
            query, answer = split_query(reverse_query)
            if mode == 1:
                mes = {"prompt": query, "completion": answer}
            else:
                mes = {"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": query}, {"role": "assistant", "content": answer}]}
            dataset.append(mes)

    return dataset



import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import login
from sklearn.model_selection import train_test_split
import os
login(os.environ["HF_TOKEN"])

sft_dataset = prepare_sft_dataset(conversation_pairs, annos, query_template, reverse_query_template, synonyms, mode=1)
df = pd.DataFrame(sft_dataset, columns=["prompt", "completion"])
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42) # Split the dataset into train and test subsets
dataset_dict = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
}) # Create a DatasetDict with train and test splits
dataset_dict.push_to_hub("Ksgk-fy/alignment-sft-test01") # Push the dataset to the Hugging Face Hub


sft_dataset = prepare_sft_dataset(conversation_pairs, annos, query_template, reverse_query_template, synonyms, mode = 1)
df = pd.DataFrame(sft_dataset, columns=["prompt", "completion"])
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
dataset_dict = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})
dataset_dict.push_to_hub("Ksgk-fy/alignment-sft-test02")

from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
import torch
from random import randint
from datasets import load_dataset
from tqdm import tqdm as tqdm
import os
import json

def format_prompt(prompt, completion):
    return f"### Question: {prompt}\n ### Answer: {completion}", f"### Question: {prompt}\n ### Answer: "

# pre-processing functional
def add_eos_dataset(dataset, tokenizer):
    eos = tokenizer.eos_token

    def add_eos(example):
        completion = example['completion']
        completion_with_eos = completion + eos
        return {'completion': completion_with_eos}

    dataset = dataset.map(add_eos)

    return dataset

def formatting_prompts_func(example):
    """
    To my current understanding, ultimately the training examples will be a bunch of text.
    """
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"### Question: {example['prompt'][i]}\n ### Answer: {example['completion'][i]}"
        output_texts.append(text)
    return output_texts

def formatting_query_prompt_func(example):
  """
  Used to let LLM generate predicted completion to a prompt
  """
  query_text = f"### Question: {example['prompt']}\n ### Answer: "
  return query_text

def format_test_dataset(dataset):
    
    def format_test_data(data):
        prompt = f"### Question: {data['prompt']}\n ### Answer: "
        return {"prompt": prompt}
    
    return dataset.map(format_test_data)


def test_lora_adaptor(base_model_id, lora_adaptor_id, dataset_id):
    
    # Corrected to use the base_model_id parameter
    model = AutoPeftModelForCausalLM.from_pretrained(
        base_model_id,  # Using the base_model_id parameter for loading the model
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)  # Using the base_model_id for the tokenizer as well
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer) # Load into pipeline with corrected variable names

    eval_dataset = load_dataset(dataset_id, split='test') # load test dataset
    eval_dataset = format_test_dataset(eval_dataset)

    # Initialize counters for statistics
    correct_answers = 0
    wrong_in_scope_answers = 0
    out_of_scope_answers = 0
    answer_lengths = 0

    for data in tqdm(eval_dataset):
        prompt = data["prompt"]
        gt_completion = data["completion"]
        outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
        pred_completion = outputs[0]['generated_text'][len(prompt):].strip()
        
        pred_first_word = pred_completion.split(' ')[0]
        answer_length = len(pred_completion.split(' '))

        if gt_completion.split(' ')[0] == pred_first_word:
            correct_answers += 1
        elif pred_first_word in ["Yes", "No"]:
            wrong_in_scope_answers += 1
        else:
            out_of_scope_answers += 1
        answer_lengths += answer_length

    # Print all statistics
    total_predictions = correct_answers + wrong_in_scope_answers + out_of_scope_answers
    avg_answer_len = answer_lengths / total_predictions if total_predictions else 0
    print(f"Total predictions made: {total_predictions}")
    print(f"Number of correct answers: {correct_answers} - Percentage: {(correct_answers / total_predictions * 100 if total_predictions else 0):.2f}%")
    print(f"Number of wrong, in-scope answers: {wrong_in_scope_answers} - Percentage: {(wrong_in_scope_answers / total_predictions * 100 if total_predictions else 0):.2f}%")
    print(f"Number of out-of-scope answers: {out_of_scope_answers} - Percentage: {(out_of_scope_answers / total_predictions * 100 if total_predictions else 0):.2f}%")
    print(f"Average length of an answer: {avg_answer_len:.2f} words")

    # Data to be written
    data = {
        "total_predictions": total_predictions,
        "correct_answers": correct_answers,
        "wrong_in_scope_answers": wrong_in_scope_answers,
        "out_of_scope_answers": out_of_scope_answers,
        "average_answer_length": avg_answer_len
    }

    return data


def save_eval_data(data, lora_adaptor_id, base_model_id, dataset_id):

    # add another key in the dict called load_adaptor_id
    data["lora_adaptor_id"] = lora_adaptor_id
    data["base_model_id"] = base_model_id
    data["dataset_id"] = dataset_id

    # Define the path for the JSON file
    json_file_path = './record/lora.json'

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

    # Check if the file exists to append or write new
    if os.path.exists(json_file_path):
        with open(json_file_path, 'a', encoding='utf-8') as file:
            # Append data to the file
            json.dump([data], file, indent=4)
    else:
        with open(json_file_path, 'w', encoding='utf-8') as file:
            # Writing data to a file
            json.dump([data], file, indent=4)



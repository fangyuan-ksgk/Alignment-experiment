from huggingface_hub import login
import peft
import trl
from peft import LoraConfig
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from trl import SFTTrainer
from tqdm import tqdm as tqdm
import numpy
import json
import argparse
import os

# Config
dataset_name = "Ksgk-fy/alignment-sft-test2-mode-1"
base_model_id = "HuggingFaceH4/zephyr-7b-beta" # base model id

# Setup argument parser
parser = argparse.ArgumentParser(description="Run LoRA alignment training.")
parser.add_argument("--bs", type=int, default=16, help="Batch size for training.")
parser.add_argument("--id", type=int, default=16, help="ID for the fine-tuned adaptor")
args = parser.parse_args()

# Set the ID for the fine-tuned adaptor
new_model_id = f"Zaligner-v1-test{args.id}"
# Retrieve batch size from arguments
batch_size = args.batch_size

print(f"Using batch size: {batch_size}")
print(f"Using model id: {new_model_id}")

HF_TOKEN = os.environ.get("HF_TOKEN")
login(
  token=HF_TOKEN, # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)

dataset = load_dataset(dataset_name, split="train")

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.padding_side = 'right' # to prevent warnings

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)

args = TrainingArguments(
    output_dir=new_model_id, # directory to save and repository id
    num_train_epochs=5,                     # number of training epochs
    per_device_train_batch_size=16,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=False,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)


def map_label(label):
    if label == "Yes":
        return "Yes"
    if label == "No":
        return "No"
    if label == "Unknown":
        return "Hmm"

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"### Question: {example['prompt'][i]}\n ### Answer: {map_label(example['completion'][i])}"
        output_texts.append(text)
    return output_texts


def formatting_prompt_func(example):
    output_texts = []
    for i in range(1):
        text = f"### Question: {example['prompt']}\n ### Answer: {map_label(example['completion'])}"
        output_texts.append(text)
    return output_texts

response_template = "### Answer:"

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


def formatting_query_prompt_func(example):
  """
  Used to let LLM generate predicted completion to a prompt
  """
  query_text = f"### Question: {example['prompt']}\n ### Answer: "
  return query_text


max_seq_length = 512 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    packing=False,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

# Evaluation on the Alignment Levels
def check_performance(dataset, model, tokenizer):
    n_correct, n_wrong = 0, 0
    pb = tqdm(total=len(dataset), desc="Calculating perplexity")
    for data in dataset:
        is_correct = eval_data(data, model, tokenizer)
        n_correct += int(is_correct)
        n_wrong += (1 - int(is_correct))
        pb.update(1)
    print("Success Rate: ", numpy.round(n_correct / (n_correct + n_wrong), 2))
    return n_correct / (n_correct + n_wrong)

def eval_data(data, model, tokenizer, scale = 1.0):

    get_text = lambda example: f"### Question: {example['prompt']}\n ### Answer: {map_label(example['completion'])}"
    chosen_str = get_text(data)
    answer_index = chosen_str.find("Answer: ") + 8
    answer = chosen_str[answer_index:].split("<|im_end|>")[0]
    start_index_sequence, end_index_sequence = answer_index, answer_index + len(answer)
    
    # When encoding happens we count the token and not the others
    query_ids = tokenizer.encode(chosen_str[:start_index_sequence])
    query_answer_ids = tokenizer.encode(chosen_str[:end_index_sequence])
    
    start_index = len(query_ids)
    end_index = len(query_answer_ids)
    if start_index == end_index:
        start_index -= 1
    
    # Run inference and calculate next-token prediction loss
    sequence_ids = tokenizer.encode(chosen_str, return_tensors="pt").to("cuda")
    with torch.no_grad():
        sequence_logits = model(sequence_ids).logits 
        target_logits = sequence_logits[:, start_index:end_index]
        # target_ids = sequence_ids[:, start_index:end_index].view(-1)
    
    # Process separately for each prediction answer
    def process_possible_answer(pos):
        pos_str = chosen_str[:start_index_sequence] + pos
        query_pos_ids = tokenizer.encode(pos_str)
        
        pos_start_index = len(query_ids)
        pos_end_index = len(query_pos_ids)
        if pos_start_index == pos_end_index:
            pos_start_index -= 1
    
        id = query_pos_ids[pos_start_index:pos_start_index+1]
        return id
    
    id_1 = process_possible_answer("Yes")
    id_2 = process_possible_answer("Hmm")
    id_3 = process_possible_answer("No")
    
    # print("Prefix: ", chosen_str[:start_index_sequence])
    # print(id_1, id_2, id_3)
    
    # Get that logits (relative logits)
    pred_logits = target_logits.view(-1)
    pred_probs = torch.softmax(pred_logits, dim=0)
    # return pred_probs, id_2

    prob_1, prob_2, prob_3 = pred_probs[id_1], pred_probs[id_2], pred_probs[id_3]

    # print(prob_1, prob_2, prob_3)
    
    prob_2 *= scale
    
    norm_1 = prob_1 / (prob_1 + prob_2 + prob_3)
    norm_2 = prob_2 / (prob_1 + prob_2 + prob_3)
    norm_3 = prob_3 / (prob_1 + prob_2 + prob_3)
    
    prob_dict = {"Yes": norm_1, "Hmm": norm_2, "No": norm_3}
    
    pred = max(prob_dict, key=prob_dict.get)
    
    print(f"Prediction: {pred}, Answer: {answer}")
    
    return pred == answer

testset = load_dataset("Ksgk-fy/alignment-sft-test2-mode-1", split="test")
success_rate = check_performance(testset, trainer.model, trainer.tokenizer)

path = "/script/logs/{new_model_id}_log.json"
info = {
    "model_id": new_model_id,
    "success_rate": success_rate
}
# Save the information to a JSON file
with open(path.format(new_model_id=new_model_id), 'w') as file:
    json.dump(info, file, indent=4)
print(f"Log saved to {path.format(new_model_id=new_model_id)}")

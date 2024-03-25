# Run this thing on Modal platform
import modal
from modal import Image, Stub, gpu
from evaluator import format_prompt
import torch
from tqdm import tqdm as tqdm
from huggingface_hub import login
import argparse
import os
# import argparse
login(os.environ["HF_TOKEN"])

# parser = argparse.ArgumentParser(description="Evaluate model perplexity.")
# parser.add_argument("--model_id", type=str, required=True, help="Model ID to evaluate.")
# parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to evaluate.")
# args = parser.parse_args()


stub = modal.Stub(
    image = Image.debian_slim(python_version="3.10")
    .pip_install(
        ["transformers", "datasets", "huggingface_hub", "torch", "tqdm"]
    )
    .apt_install("git")
    .apt_install( "gcc")
    .run_commands(f"export HF_TOKEN={os.environ['HF_TOKEN']}")
    .run_commands("export WANDB_API_KEY=0a22c2f6b4be0592a7867dbc40d6e83fcfdd305a")
    .run_commands("git config --global user.name ksgk-fangyuan",
                  "git config --global user.email fangyuan.yu18@gmail.com",
                  )
)

@stub.function(gpu = modal.gpu.A100(size="40GB"))
def evaluate_perplexity(model_id, dataset_name):

    import argparse, datasets, transformers
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    
    evaluate_perplexity(model, tokenizer, dataset)

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset(dataset_name, split="test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Updated to use CPU as fallback
    model.to(device)
    model.eval()

    perplexities = []

    for data in tqdm(dataset, desc="Evaluating Perplexity on the Instruction-Tuning Dataset"):
        prompt = data["prompt"]
        completion = data["completion"] + tokenizer.eos_token

        target_sequence, query_sequence = format_prompt(prompt, completion)
        query_sequence_length = tokenizer.encode(query_sequence, return_tensors="pt").shape[1]

        sequence_ids = tokenizer.encode(target_sequence, return_tensors="pt").to(device)  # Ensure sequence_ids are on the correct device

        with torch.no_grad():
            sequence_logits = model(sequence_ids).logits
            target_logits = sequence_logits[:, (query_sequence_length-1):-1]
            target_ids = sequence_ids[:, query_sequence_length:].view(-1)

        target_logits = sequence_logits[:, (query_sequence_length-1):-1]
        target_ids = sequence_ids[:, query_sequence_length:].view(-1)

        loss = torch.nn.functional.cross_entropy(target_logits.reshape(-1, target_logits.size(-1)), target_ids, reduction="none")

        perplexity = loss.mean()
        perplexities.append(perplexity)

    return sum(perplexities) / len(perplexities)



@stub.local_entrypoint()
def main():
    parser = argparse.ArgumentParser(description="Evaluate model perplexity on a dataset")
    parser.add_argument("--model_id", type=str, default="HuggingFaceH4/zephyr-7b-beta", required=False, help="The model identifier from Huggingface's Model Hub")
    parser.add_argument("--dataset_name", type=str, default="Ksgk-fy/alignment-sft-test01", required=False, help="The name of the dataset to evaluate on")
    args = parser.parse_args()

    evaluate_perplexity.remote(args.model_id, args.dataset_name)


# if __name__ == "__main__":
#     main()

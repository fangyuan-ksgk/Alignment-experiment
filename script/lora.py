# Fine-Tuning with LoRA (??)
import modal
from modal import Image, Stub, gpu
import torch
from tqdm import tqdm
from huggingface_hub import login
import argparse
import os
from util import *

login(os.environ["HF_TOKEN"])

# Axolotl image hash corresponding to 0.4.0 release (2024-02-14)
AXOLOTL_REGISTRY_SHA = (
    "d5b941ba2293534c01c23202c8fc459fd2a169871fa5e6c45cb00f363d474b6a"
)


stub = modal.Stub(
    image = (Image.from_registry(f"winglian/axolotl@sha256:{AXOLOTL_REGISTRY_SHA}")
        .run_commands(
            "git clone https://github.com/OpenAccess-AI-Collective/axolotl /root/axolotl",
            "cd /root/axolotl && git checkout v0.4.0",
        )
    )
    # image = Image.debian_slim(python_version="3.11")
    .pip_install(
        ["transformers", "datasets", "huggingface_hub", "torch", "tqdm", "psutil", "sentencepiece", "peft", "accelerate", "bitsandbytes", "trl"]
    )
    .apt_install("git")
    .apt_install( "gcc")
    .run_commands("git config --global user.name ksgk-fangyuan",
                  "git config --global user.email fangyuan.yu18@gmail.com",
                  )
)
@stub.function(gpu = modal.gpu.A100(size="40GB"),
               secrets=[modal.Secret.from_name("ksgk-secret")],
               timeout=24000)
def lora_finetune_01(base_model_id, lora_adaptor_id, dataset_id, epochs=3):

    from huggingface_hub import login
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    from peft import LoraConfig
    import trl
    from peft import LoraConfig
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer

    login(os.environ["HF_TOKEN"])

    # Load & Preprocess Dataset
    dataset = load_dataset(dataset_id, split="train")

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

    # Preprocess Dataset
    dataset = add_eos_dataset(dataset, tokenizer)

    # LoRA Config -- this could be another variable here
    # Worth Digging into Sebastian Raschka's experiment and see what are the tricks for the LoRA fine-tuning 
    # -- might be some funny stuff that I am missing here ....
    peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=256,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
    )

    # Some Training Arguments
    args = TrainingArguments(
        output_dir="alignment-adaptor-test01", # directory to save and repository id
        num_train_epochs=epochs,                     # number of training epochs
        per_device_train_batch_size=16,          # batch size per device during training
        gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-4,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=True,                       # push model to hub
        report_to="tensorboard",                # report metrics to tensorboard
    )

    max_seq_length = 512

    trainer = SFTTrainer(
        model = model,
        args=args,
        train_dataset = dataset,
        peft_config = peft_config,
        max_seq_length = max_seq_length,
        tokenizer = tokenizer,
        formatting_func = formatting_prompts_func,
        dataset_kwargs = {
            "add_special_tokens": False,  # We teplate with special tokens
            "append_concat_token": False, # No need to add additional separator token
        }
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model()

    # release memory
    del model 
    del trainer

    # Evaluate model & Store example inference result
    data = test_lora_adaptor(base_model_id, lora_adaptor_id, dataset_id)

    return data


@stub.local_entrypoint()
def main(base_model_id = "HuggingFaceH4/zephyr-7b-beta",
         lora_adaptor_id = "Ksgk-fy/alignment-sft-test01",
         dataset_id = "Ksgk-fy/alignment-sft-test01",
         epochs = 3):
    
    if isinstance(epochs, str):
        epochs = int(epochs)
    
    # Fine-Tune & Evaluate
    data = lora_finetune_01.remote(base_model_id, lora_adaptor_id, dataset_id, epochs=epochs)

    # Save Evaluation Data
    save_eval_data(data, lora_adaptor_id, base_model_id, dataset_id)

    print("LoRA Fine-Tuning version: {lora_adaptor_id} completed")




















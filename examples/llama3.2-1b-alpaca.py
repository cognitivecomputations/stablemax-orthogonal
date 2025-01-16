#!/usr/bin/env python
# examples/llama3.2-1b-alpaca.py

import logging
import os
from typing import Dict

import datasets
from datasets import load_dataset
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling

# Import your custom modules from src/
# Adjust the relative import as needed depending on your project structure
from src import (
    CustomTrainingArguments,
    CustomTrainer,
    StableMax,
    LogStableMax,
)

logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------------------------
# 1. Define special tokens and prompt formatting
# -------------------------------------------------------------------------

# For demonstration, we define some placeholder special tokens.
# Adjust to your actual special tokens if your tokenizer already has them.
SPECIAL_TOKENS = {
    "begin_of_text": "<|begin_of_text|>",
    "start_header_id": "<|start_header_id|>",
    "end_header_id": "<|end_header_id|>",
    "eot_id": "<|eot_id|>",
}

# Construct the full prompt format according to your instructions
# Alpaca data: {"instruction": "...", "input": "...", "output": "..."}
# Desired prompt:
# <|begin_of_text|><|start_header_id|>user<|end_header_id|>
# Given the following instruction: <instruction> and the following input: <input> please generate a response
# <|eot_id|><|start_header_id|>assistant<|end_header_id|>

PROMPT_TEMPLATE = (
    "{begin_of_text}{start_header_id}user{end_header_id}\n"
    "Given the following instruction: {instruction} and the following input: {input} please generate a response\n"
    "{eot_id}{start_header_id}assistant{end_header_id}\n"
)

def build_prompt(instruction: str, input_text: str) -> str:
    """
    Returns the prompt string based on instruction and input_text.
    """
    # If there's no input, we can handle that case as well (Alpaca often has empty inputs).
    # Customize as needed.
    user_input = input_text if input_text.strip() else "N/A"
    return PROMPT_TEMPLATE.format(
        begin_of_text=SPECIAL_TOKENS["begin_of_text"],
        start_header_id=SPECIAL_TOKENS["start_header_id"],
        end_header_id=SPECIAL_TOKENS["end_header_id"],
        eot_id=SPECIAL_TOKENS["eot_id"],
        instruction=instruction,
        input=user_input,
    )

# -------------------------------------------------------------------------
# 2. Data processing & tokenization
# -------------------------------------------------------------------------

def tokenize_example(example: Dict, tokenizer: AutoTokenizer, max_length: int = 1024) -> Dict:
    """
    Tokenize an Alpaca example:
      example = {"instruction": "...", "input": "...", "output": "..."}

    We'll build the prompt, then the labels will be the entire sequence
    plus the output.  The 'output' portion can be appended with
    a newline or special token if desired.
    """
    instruction = example["instruction"]
    input_text = example["input"]
    output_text = example["output"]  # The model should generate this text.

    # Build the user prompt
    prompt_str = build_prompt(instruction, input_text)

    # Full conversation for training = prompt_str + output_text
    full_text = prompt_str + output_text

    # Tokenize
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="longest",  # or "max_length" if you prefer
    )

    # We want the model to predict the entire sequence. 
    # Usually for causal LM training, 'labels' = input_ids. 
    # For partial context masking, you'd do something else.
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

# -------------------------------------------------------------------------
# 3. Main script: load model, dataset, trainer, and train
# -------------------------------------------------------------------------

def main():
    # HF Hub model name for the Llama-3.2-1B weights
    model_name = "meta-llama/Llama-3.2-1B"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Optionally, add or define special tokens, if they are not already in the tokenizer.
    # If your Llama tokenizer already covers these tokens, skip this step or adapt as needed.
    special_tokens_list = list(SPECIAL_TOKENS.values())
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_list})

    # Load the Llama model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # Use BF16 or other precision if your hardware supports it
        torch_dtype=torch.bfloat16,
        device_map="auto",  # or 'cuda:0' or 'cpu', depending on your setup
    )

    # Because we added special tokens, we may need to resize the token embeddings
    # for the new tokens we just added. 
    model.resize_token_embeddings(len(tokenizer))

    # Load the alpaca-cleaned dataset
    dataset_name = "yahma/alpaca-cleaned"
    raw_dataset = load_dataset(dataset_name)

    # We'll do a train/validation split. 
    # 'train' is the entire dataset, so let's do a small fraction for val if not provided.
    if "validation" not in raw_dataset:
        # We'll split train into 90%/10% if no separate validation is provided.
        raw_dataset = raw_dataset["train"].train_test_split(test_size=0.1)
        train_dataset = raw_dataset["train"]
        val_dataset   = raw_dataset["test"]
    else:
        train_dataset = raw_dataset["train"]
        val_dataset   = raw_dataset["validation"]

    # Tokenize the dataset
    def tokenize_fn(examples):
        return tokenize_example(examples, tokenizer, max_length=2048)  # Llama has large context

    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=False,  # each example is processed individually
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        tokenize_fn,
        batched=False,
        remove_columns=val_dataset.column_names,
    )

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # for causal LM
    )

    # Prepare custom training arguments
    training_args = CustomTrainingArguments(
        output_dir="./llama3.2-1b-alpaca-checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=1,  # adjust as needed
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        learning_rate=2e-5,
        # BF16 can be used if GPU supports it
        bf16=True,
        deepspeed=None,  # Optional if you want DeepSpeed config
        # Enable stablemax or log_stablemax if desired
        use_stable_max=True,  # or False if you only want standard softmax
        use_log_stable_max=False,
        # Use orthogonal gradient optimizer
        use_orthogonal_optimizer=True,
        skip_orthogonal_param_types=["bias", "LayerNorm"],
        # The LlamaForCausalLM mapping likely expects "lm_head" for the final layer
        stable_max_layer_name="auto",
        # Possibly expand final layer if desired (experimental)
        expand_final_layer=False,
        # Normal HF Trainer args
        logging_dir="./logs/llama3.2-1b-alpaca",
        report_to=["tensorboard"],
        do_train=True,
        do_eval=True,
    )

    # Instantiate our custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Evaluate
    eval_metrics = trainer.evaluate()
    print("Evaluation:", eval_metrics)

    # Save final model
    trainer.save_model("./llama3.2-1b-alpaca-final")

    print("All done! Model checkpoints saved to:", training_args.output_dir)

if __name__ == "__main__":
    main()

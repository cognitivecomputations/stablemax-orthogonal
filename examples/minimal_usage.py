# examples/minimal_usage.py

import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset

# Import our custom classes/functions from the src package
from src import (
    StableMax,
    LogStableMax,
    with_orthogonal_gradient,
    MODEL_CONFIG_MAPPING,
    CustomTrainingArguments,
    CustomTrainer,
)

logging.basicConfig(level=logging.INFO)


def main():
    """
    A minimal usage example demonstrating how to:
      1. Load a small model and tokenizer
      2. Load a small dataset
      3. Create custom training arguments
      4. Instantiate our CustomTrainer and train
    """

    # 1. Choose a small model for quick tests (e.g., a tiny GPT-2)
    model_name = "sshleifer/tiny-gpt2"  # or "gpt2", "meta-llama/Llama-2-7b-hf", etc.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Some tokenizers may not have a pad token; if so, add one
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # 2. Load a small dataset
    #    Here we use the "wikitext" dataset (wikitext-2-raw-v1) as a quick example.
    #    For real-world use, replace with your own dataset or another HF dataset.
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    # We'll just do a quick demonstration on a few samples
    small_train_dataset = dataset["train"].select(range(128))     # Take a small subset
    small_val_dataset   = dataset["validation"].select(range(64)) # Another small subset

    # Collator to handle text batching
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 3. Create custom training arguments
    #    Try turning on `use_stable_max` or `use_log_stable_max` for the final activation,
    #    and `use_orthogonal_optimizer` for orthogonal gradient decomposition.
    training_args = CustomTrainingArguments(
        output_dir="outputs/minimal_example",
        overwrite_output_dir=True,
        num_train_epochs=1,              # For demonstration
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=1,
        learning_rate=1e-4,
        logging_dir="logs/minimal_example",
        logging_steps=10,
        # Enable one or both of these to test StableMax/LogStableMax
        use_stable_max=True,
        use_log_stable_max=False,
        # Enable orthogonal gradient decomposition
        use_orthogonal_optimizer=True,
        skip_orthogonal_param_types=["bias", "LayerNorm"],  # Example skip list
        # For automatic final-layer detection
        stable_max_layer_name="auto",
        # Whether or not to expand the final layer by +1 dimension
        expand_final_layer=False,
        # (Optional) specify other HF arguments, e.g. seed=42, fp16=True, etc.
    )

    # 4. Instantiate our CustomTrainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_val_dataset,
        data_collator=data_collator,
    )

    # 5. Train the model
    trainer.train()

    # Optionally evaluate or save
    trainer.evaluate()
    trainer.save_model("outputs/minimal_example/final_model")

    print("Training complete!")


if __name__ == "__main__":
    main()

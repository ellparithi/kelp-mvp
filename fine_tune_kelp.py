
import os
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

def fine_tune_kelp(kelp_name, base_model="microsoft/phi-1_5", output_dir=None, epochs=3):
    """
    Fine-tune a Huggingface model on the raw Kelp memory (CPU optimized).
    """

    raw_data_path = f"kelpdata/{kelp_name}/raw_docs.txt"
    model_output_path = output_dir or f"kelpmodels/{kelp_name}"

    if not os.path.exists(raw_data_path):
        print(f"❌ No raw data found at {raw_data_path}")
        return

    print(f"✅ Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Keep lightweight dtype
        device_map="cpu"            # Force CPU training
    )

    print(f"✅ Preparing dataset from {raw_data_path}")
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    dataset = Dataset.from_dict({"text": raw_lines})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256  # smaller block size for memory
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    print(f"✅ Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir=model_output_path,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,  # smaller batch
        save_steps=10,
        save_total_limit=2,
        logging_steps=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        report_to="none",
        no_cuda=True,  # ✅ Important: Disable CUDA/MPS, force CPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    print(f"🏋️‍♂️ Fine-tuning model...")
    trainer.train()

    print(f"✅ Saving model to {model_output_path}")
    model.save_pretrained(model_output_path)
    tokenizer.save_pretrained(model_output_path)

    print(f"🎯 Fine-tuning complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kelp_name", required=True, help="Name of the Kelp collection")
    parser.add_argument("--base_model", default="microsoft/phi-1_5", help="Base Huggingface model")
    parser.add_argument("--output_dir", default=None, help="Optional output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")

    args = parser.parse_args()

    fine_tune_kelp(
        kelp_name=args.kelp_name,
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs
    )

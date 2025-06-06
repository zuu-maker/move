import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
import evaluate
from rouge_score import rouge_scorer
import numpy as np
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from tqdm import tqdm

# 1. Load bridge dataset
print("Loading bridge inspection dataset...")
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Load your bridge CSV
df = pd.read_csv('dataset_new.csv')

print(f"Loaded {len(df)} bridge components from {df['bridge_id'].nunique()} bridges")

# Create train/validation/test splits
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

# Convert to HuggingFace Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# FIXED: Create dataset dict with all three splits
dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,  # Proper validation set
    'test': test_dataset        # Actual test set for final evaluation
})

print("Loading model and tokenizer...")
model_name = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=config
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)


def preprocess_function(examples):
    # Bridge format for Phi-2 (causal LM)
    full_texts = [f"### Bridge Assessment: {q}\n### Answer: {a}"
                  for q, a in zip(examples["question"], examples["answer"])]

    # Tokenize the full texts - REDUCED MAX LENGTH FOR MEMORY
    tokenized = tokenizer(full_texts, padding=True, truncation=True, max_length=256)  # Reduced from 512

    # Set up the model inputs
    result = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"].copy()  # For causal LM, labels = input_ids
    }

    return result


# Process datasets - FIXED to use all three splits
train_dataset = dataset["train"].map(preprocess_function, batched=True)
eval_dataset = dataset["validation"].map(preprocess_function, batched=True)  # Validation for during training
test_dataset = dataset["test"].map(preprocess_function, batched=True)  # Test for final evaluation

# Load evaluation metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
bertscore_metric = evaluate.load("bertscore")


# ADD: Function to extract condition rating from text
def extract_condition_rating(text):
    """Extract condition rating from generated or reference text"""
    # Define possible condition ratings
    condition_ratings = ['GOOD', 'SATISFACTORY', 'FAIR', 'POOR', 'SEVERE', 'CRITICAL']

    # Convert text to uppercase for matching
    text_upper = text.upper()

    # Look for condition ratings in the text
    for rating in condition_ratings:
        if rating in text_upper:
            return rating

    # Try to find variations
    if any(word in text_upper for word in ['EXCELLENT', 'VERY GOOD']):
        return 'GOOD'
    elif any(word in text_upper for word in ['ACCEPTABLE', 'ADEQUATE']):
        return 'SATISFACTORY'
    elif any(word in text_upper for word in ['DETERIORATED', 'DEGRADED']):
        return 'POOR'
    elif any(word in text_upper for word in ['CRITICAL', 'URGENT', 'IMMEDIATE']):
        return 'SEVERE'

    return 'UNKNOWN'


# ADD: Function to compute classification metrics
def compute_classification_metrics(pred_ratings, true_ratings):
    """Compute classification metrics for condition ratings"""
    # Filter out UNKNOWN predictions
    valid_indices = [(i, p, t) for i, (p, t) in enumerate(zip(pred_ratings, true_ratings))
                     if p != 'UNKNOWN' and t != 'UNKNOWN']

    if not valid_indices:
        return {
            'condition_accuracy': 0.0,
            'condition_precision': 0.0,
            'condition_recall': 0.0,
            'condition_f1': 0.0,
            'condition_unknown_rate': 1.0
        }

    valid_preds = [p for _, p, _ in valid_indices]
    valid_trues = [t for _, _, t in valid_indices]

    # Calculate metrics
    accuracy = accuracy_score(valid_trues, valid_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_trues, valid_preds, average='weighted', zero_division=0
    )

    unknown_rate = 1.0 - (len(valid_indices) / len(pred_ratings))

    return {
        'condition_accuracy': accuracy,
        'condition_precision': precision,
        'condition_recall': recall,
        'condition_f1': f1,
        'condition_unknown_rate': unknown_rate
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Replace -100 in labels with pad token id for decoding
    labels_cleaned = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and references
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels_cleaned, skip_special_tokens=True)

    # Extract just the response part for cleaner comparison
    extracted_preds = []
    extracted_refs = []

    # ADD: Lists for condition ratings
    pred_condition_ratings = []
    true_condition_ratings = []

    for pred, label in zip(decoded_preds, decoded_labels):
        # Simple extraction of the response after "### Response: "
        if "### Response: " in pred:
            pred_response = pred.split("### Response: ")[-1].strip()
            extracted_preds.append(pred_response)
            # Extract condition rating from prediction
            pred_condition_ratings.append(extract_condition_rating(pred_response))
        else:
            extracted_preds.append(pred.strip())
            pred_condition_ratings.append(extract_condition_rating(pred.strip()))

        if "### Response: " in label:
            label_response = label.split("### Response: ")[-1].strip()
            extracted_refs.append([label_response])  # BLEU needs a list of references for each prediction
            # Extract condition rating from true label
            true_condition_ratings.append(extract_condition_rating(label_response))
        else:
            extracted_refs.append([label.strip()])
            true_condition_ratings.append(extract_condition_rating(label.strip()))

    # For non-BLEU metrics, we need single references, not lists
    flat_refs = [ref[0] for ref in extracted_refs]

    try:
        # Calculate BLEU score using evaluate
        bleu_results = bleu_metric.compute(predictions=extracted_preds, references=extracted_refs)

        # Calculate ROUGE scores
        rouge_results = rouge_metric.compute(
            predictions=extracted_preds,
            references=flat_refs,
            use_stemmer=True
        )

        # Calculate METEOR score
        meteor_results = meteor_metric.compute(
            predictions=extracted_preds,
            references=flat_refs
        )

        # Calculate BERTScore (semantic similarity) - SIMPLIFIED FOR MEMORY
        try:
            bertscore_results = bertscore_metric.compute(
                predictions=extracted_preds[:10],  # Only first 10 for memory
                references=flat_refs[:10],
                lang="en",
                model_type="distilbert-base-uncased"
            )
            bert_precision = np.mean(bertscore_results["precision"])
            bert_recall = np.mean(bertscore_results["recall"])
            bert_f1 = np.mean(bertscore_results["f1"])
        except Exception as e:
            print(f"Warning: BERTScore calculation failed: {e}")
            bert_precision = bert_recall = bert_f1 = 0

        # Calculate exact match ratio
        exact_match_ratio = sum(1 for p, r in zip(extracted_preds, flat_refs) if p.strip() == r.strip()) / len(
            extracted_preds)

        # ADD: Calculate condition classification metrics
        condition_metrics = compute_classification_metrics(pred_condition_ratings, true_condition_ratings)

        # ADD: Print sample predictions for debugging
        if len(pred_condition_ratings) > 0:
            print("\nSample condition predictions:")
            for i in range(min(3, len(pred_condition_ratings))):
                print(f"  Pred: {pred_condition_ratings[i]}, True: {true_condition_ratings[i]}")

        # Return all metrics including condition classification metrics
        return {
            "bleu": bleu_results["bleu"],
            "rouge1": rouge_results["rouge1"],
            "rouge2": rouge_results["rouge2"],
            "rougeL": rouge_results["rougeL"],
            "meteor": meteor_results["meteor"],
            "bertscore_precision": bert_precision,
            "bertscore_recall": bert_recall,
            "bertscore_f1": bert_f1,
            "exact_match_ratio": exact_match_ratio,
            # ADD: Condition classification metrics
            "condition_accuracy": condition_metrics['condition_accuracy'],
            "condition_precision": condition_metrics['condition_precision'],
            "condition_recall": condition_metrics['condition_recall'],
            "condition_f1": condition_metrics['condition_f1'],
            "condition_unknown_rate": condition_metrics['condition_unknown_rate'],
        }

    except Exception as e:
        print(f"Metrics computation failed: {e}")
        return {
            "bleu": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "meteor": 0.0,
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
            "exact_match_ratio": 0.0,
            # ADD: Default condition metrics
            "condition_accuracy": 0.0,
            "condition_precision": 0.0,
            "condition_recall": 0.0,
            "condition_f1": 0.0,
            "condition_unknown_rate": 1.0,
        }


# MEMORY OPTIMIZED TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="test",
    per_device_train_batch_size=1,  # REDUCED FROM 2
    per_device_eval_batch_size=1,  # REDUCED FOR MEMORY
    gradient_accumulation_steps=4,  # MAINTAIN EFFECTIVE BATCH SIZE
    num_train_epochs=1,  # FULL EPOCHS INSTEAD OF MAX_STEPS
    # learning_rate=5e-4,  # EXPLICIT LEARNING RATE
    warmup_steps=50,
    # save_strategy="epoch",  # SAVE LESS FREQUENTLY
    logging_steps=50,  # LOG LESS FREQUENTLY
    # eval_strategy="epoch",  # EVAL LESS FREQUENTLY
    report_to="none",
    # num_train_epochs=1,  # Reduce from 3 to 1
    learning_rate=1e-4,  # Lower learning rate
    weight_decay=0.01,  # Add regularization
    warmup_ratio=0.1,  # More warmup
    save_strategy="steps",
    eval_strategy="steps",
    eval_steps=50,  # Evaluate more frequently
    remove_unused_columns=True,
    fp16=True,  # MEMORY OPTIMIZATION
    gradient_checkpointing=True,  # MEMORY OPTIMIZATION
    dataloader_pin_memory=False,  # REDUCE MEMORY PRESSURE
    eval_accumulation_steps=1,  # PROCESS EVAL IN SMALLER CHUNKS
    save_total_limit=1,  # KEEP ONLY 1 CHECKPOINT
    load_best_model_at_end=False  # DISABLE TO SAVE MEMORY
)

# Create trainer with metrics and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Validation set for during training
    compute_metrics=compute_metrics,
    tokenizer=tokenizer  # PASS TOKENIZER FOR BETTER HANDLING
)

print("Starting bridge inspection training with memory optimizations...")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Clear cache before training
torch.cuda.empty_cache()

# Train the model
trainer.train()

# IMPORTANT: Final evaluation on the UNSEEN TEST SET
print("\n" + "="*50)
print("Running FINAL EVALUATION on UNSEEN TEST SET...")
print("="*50)

# Evaluate on the test set that was never seen during training
test_results = trainer.evaluate(eval_dataset=test_dataset)

print(f"\nTest Set Results (Unseen Data):")
for key, value in test_results.items():
    print(f"  {key}: {value:.4f}")

# Print condition classification results specifically
print("\nCondition Classification Results on Test Set:")
print(f"  Accuracy: {test_results.get('eval_condition_accuracy', 0):.3f}")
print(f"  Precision: {test_results.get('eval_condition_precision', 0):.3f}")
print(f"  Recall: {test_results.get('eval_condition_recall', 0):.3f}")
print(f"  F1 Score: {test_results.get('eval_condition_f1', 0):.3f}")
print(f"  Unknown Rate: {test_results.get('eval_condition_unknown_rate', 0):.3f}")

# Optional: Also show validation set results for comparison
print("\n" + "="*50)
print("Validation Set Results (seen during training):")
val_results = trainer.evaluate(eval_dataset=eval_dataset)
for key, value in val_results.items():
    print(f"  {key}: {value:.4f}")

# Test generation on a sample
print("\n" + "="*50)
print("Testing generation on a sample...")
test_input = "### Bridge Assessment: Component: Steel Girder. Defects: Minor corrosion. Assess condition.\n### Answer:"
inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_length=inputs['input_ids'].shape[1] + 50,
        num_beams=2,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Input: {test_input}")
print(f"Generated: {generated_text}")

# Save the model
print("\nSaving the fine-tuned model...")
model.save_pretrained("./bridge-inspection-phi-1_5-lora")
tokenizer.save_pretrained("./bridge-inspection-phi-1_5-lora")
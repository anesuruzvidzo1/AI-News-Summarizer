
import os
import argparse
import random
from typing import List, Dict
from pymongo import MongoClient
from tqdm import tqdm
import csv
import numpy as np
import torch

from datasets import Dataset, DatasetDict
import evaluate

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# ---------- Default fields (change here if your processed collection uses different keys) ----------
SRC_FIELD = "clean_text"
TGT_FIELD = "summary"

# ---------- Mongo fetch ----------
def fetch_processed_from_mongo(mongo_uri: str, db_name: str, collection_name: str, require_summary: bool = True):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    col = db[collection_name]
    query = {}
    if require_summary:
        query[TGT_FIELD] = {"$exists": True, "$ne": ""}
    projection = {SRC_FIELD: 1, TGT_FIELD: 1, "_id": 0}
    cursor = col.find(query, projection)
    records = []
    for doc in tqdm(cursor, desc="Fetching processed docs from Mongo"):
        src = doc.get(SRC_FIELD) or ""
        tgt = doc.get(TGT_FIELD) or ""
        # Basic sanity checks
        if not src or not tgt:
            continue
        src = src.strip()
        tgt = tgt.strip()
        if len(src) < 30 or len(tgt) < 5:
            continue
        records.append({"article": src, "summary": tgt})
    return records

# ---------- Build dataset ----------
def build_dataset(records: List[Dict], val_fraction: float = 0.1, seed: int = 42):
    if not records:
        raise ValueError("No records found. Ensure processed_articles contains documents with 'clean_text' and 'summary'.")
    random.Random(seed).shuffle(records)
    cutoff = int(len(records) * (1 - val_fraction))
    train = records[:cutoff]
    val = records[cutoff:]
    ds = DatasetDict({"train": Dataset.from_list(train), "validation": Dataset.from_list(val)})
    return ds

# ---------- Tokenize ----------
def tokenize_and_prepare(ds: DatasetDict, tokenizer, max_source_length: int, max_target_length: int):
    def preprocess_batch(examples):
        inputs = examples["article"]
        targets = examples["summary"]
        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
        model_inputs["labels"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        return model_inputs

    tokenized_train = ds["train"].map(preprocess_batch, batched=True, remove_columns=ds["train"].column_names)
    tokenized_valid = ds["validation"].map(preprocess_batch, batched=True, remove_columns=ds["validation"].column_names)
    return tokenized_train, tokenized_valid

# ---------- Metrics ----------
def compute_metrics_factory(tokenizer):
    rouge_metric = evaluate.load("rouge")
    def postprocess_text(preds, labels):
        preds = [p.strip() for p in preds]
        labels = [l.strip() for l in labels]
        return preds, labels

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        formatted = {}
        for key, val in result.items():
            if isinstance(val, dict) and "fmeasure" in val:
                formatted[key] = round(val["fmeasure"] * 100, 4)
            else:
                try:
                    formatted[key] = round(float(val) * 100, 4)
                except Exception:
                    formatted[key] = val
        try:
            prediction_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]
            formatted["gen_len"] = float(np.mean(prediction_lens)) if len(prediction_lens) > 0 else 0.0
        except Exception:
            formatted["gen_len"] = 0.0
        return formatted
    return compute_metrics

# ---------- Save predictions ----------
def save_validation_predictions(model, tokenizer, dataset, output_csv: str, max_length: int = 128, num_beams: int = 4):
    rows = []
    for i, example in enumerate(tqdm(dataset, desc="Generating validation predictions")):
        src = example["article"]
        inputs = tokenizer(src, truncation=True, max_length=512, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        out = model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=True)
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        ref = example["summary"]
        rows.append({"id": i, "source": src, "prediction": pred, "reference": ref})
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "source", "prediction", "reference"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mongo_uri", type=str, default="mongodb://localhost:27017/")
    p.add_argument("--db_name", type=str, default="news_db")
    p.add_argument("--collection", type=str, default="processed_articles")
    p.add_argument("--model_name_or_path", type=str, default="t5-small")
    p.add_argument("--output_dir", type=str, default="./outputs/t5_from_mongo")
    p.add_argument("--max_source_length", type=int, default=512)
    p.add_argument("--max_target_length", type=int, default=128)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--save_preds", type=str, default="validation_preds.csv")
    return p.parse_args()

# ---------- Main ----------
import argparse
def main():
    args = parse_args()
    set_seed(args.seed)

    print("Fetching processed records from MongoDB...")
    records = fetch_processed_from_mongo(args.mongo_uri, args.db_name, args.collection, require_summary=True)
    print(f"Found {len(records)} usable records (with summaries).")
    if len(records) < 10:
        raise SystemExit("Not enough records to train (need >=10). Ensure processed_articles has labeled summaries.")

    ds = build_dataset(records, val_fraction=args.val_fraction, seed=args.seed)
    print(f"Dataset sizes → train: {len(ds['train'])}, validation: {len(ds['validation'])}")

    print("Loading tokenizer & model:", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)

    if torch.cuda.is_available():
        model = model.cuda()

    print("Tokenizing dataset...")
    tokenized_train, tokenized_valid = tokenize_and_prepare(ds, tokenizer, args.max_source_length, args.max_target_length)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    compute_metrics = compute_metrics_factory(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        save_total_limit=3,
        fp16=args.fp16 and torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model & tokenizer to:", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Generating validation predictions and saving to CSV...")
    try:
        best = trainer.state.best_model_checkpoint
        if best:
            model = AutoModelForSeq2SeqLM.from_pretrained(best)
            if torch.cuda.is_available():
                model = model.cuda()
    except Exception:
        pass

    save_validation_predictions(model, tokenizer, ds["validation"], args.save_preds, max_length=args.max_target_length)
    print("Saved validation predictions to", args.save_preds)

    print("Running final evaluation on validation set...")
    final_metrics = trainer.evaluate()
    print("\n===== FINAL ROUGE METRICS =====")
    for k, v in final_metrics.items():
        print(f"{k}: {v}")
    print("Done.")

if __name__ == "__main__":
    main()

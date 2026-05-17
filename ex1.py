import argparse
import os
import numpy as np
import wandb

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "mrpc"
NUM_LABELS = 2

RES_FILE = "res.txt"
PREDICTIONS_FILE = "predictions.txt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_and_prepare_data(max_train, max_eval, max_predict):
    raw = load_dataset(DATASET_NAME, DATASET_CONFIG)

    if max_train != -1:
        raw["train"] = raw["train"].select(range(max_train))
    if max_eval != -1:
        raw["validation"] = raw["validation"].select(range(max_eval))
    if max_predict != -1:
        raw["test"] = raw["test"].select(range(max_predict))

    return raw


def tokenize_dataset(raw, tokenizer):
    def tokenize(example):
        return tokenizer(
            example["sentence1"],
            example["sentence2"],
            truncation=True,
        )

    tokenized = raw.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized = tokenized.rename_column("label", "labels")
    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


def append_res_line(num_epochs, lr, batch_size, eval_acc, path=RES_FILE):
    with open(path, "a") as f:
        f.write(
            f"epoch_num: {num_epochs}, lr: {lr}, "
            f"batch_size: {batch_size}, eval_acc: {eval_acc:.4f}\n"
        )


def train_model(args, tokenized, tokenizer):
    set_seed(args.seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    run_name = f"lr_{args.lr}_bs_{args.batch_size}_ep_{args.num_train_epochs}"
    output_dir = os.path.join("output", run_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="wandb",
        run_name=run_name,
        seed=args.seed,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    eval_acc = eval_results["eval_accuracy"]

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    append_res_line(args.num_train_epochs, args.lr, args.batch_size, eval_acc)

    wandb.finish()

    return eval_acc, output_dir


def predict(args, raw, tokenized, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="output_pred",
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=collator,
    )

    output = trainer.predict(tokenized["test"])
    preds = np.argmax(output.predictions, axis=-1)

    with open(PREDICTIONS_FILE, "w") as f:
        for i, label in enumerate(preds):
            s1 = raw["test"][i]["sentence1"]
            s2 = raw["test"][i]["sentence2"]
            f.write(f"{s1}###{s2}###{int(label)}\n")

    return preds


def main():
    args = parse_args()
    set_seed(args.seed)

    base_tokenizer_path = (
        args.model_path if (args.do_predict and not args.do_train) else MODEL_NAME
    )
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)

    raw = load_and_prepare_data(
        args.max_train_samples, args.max_eval_samples, args.max_predict_samples
    )
    tokenized = tokenize_dataset(raw, tokenizer)

    if args.do_train:
        train_model(args, tokenized, tokenizer)

    if args.do_predict:
        if args.model_path is None:
            raise ValueError("--model_path is required when --do_predict is set.")
        predict(args, raw, tokenized, tokenizer)


if __name__ == "__main__":
    main()

import torch
import os
import pandas as pd
import config
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

from Translator.data_parser import parse_xml_to_csv
from Translator.dataset import ElvishDataset
from Translator.metrics import compute_metrics

def train_model():

    if not os.path.exists(config.CSV_FILE):
        parse_xml_to_csv(config.XML_FILE, config.CSV_FILE)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    df = pd.read_csv(config.CSV_FILE)

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_dataset = ElvishDataset(train_df, tokenizer, prefix=config.MODEL_PREFIX)
    eval_dataset = ElvishDataset(test_df, tokenizer, prefix=config.MODEL_PREFIX)

    args = Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        save_total_limit=2,
        num_train_epochs=config.EPOCHS,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=lambda preds: compute_metrics(preds, tokenizer)
    )

    print("Starting training")
    trainer.train()

    print(f"Saving model to {config.OUTPUT_DIR}")
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)


if __name__ == "__main__":
    train_model()
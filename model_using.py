import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import config
import argparse

def load_model(model_path):
    if not os.path.isdir(model_path):
        raise SystemExit(
            f"No trained model found at '{model_path}'. "
            "Run 'python train_model.py' first to create it."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

def translate(text, model, tokenizer, device):
    words = text.split()
    translated_words = []
    for word in words:
        input_text = config.MODEL_PREFIX + word
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        outputs = model.generate(
            inputs.input_ids,
            max_length=40,
            num_beams=4,
            early_stopping=True
        )
        decoded_words = tokenizer.decode(outputs[0], skip_special_tokens=True)
        clean_word = decoded_words.rstrip("0123456789")
        translated_words.append(clean_word)

    return " ".join(translated_words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Text to translate")
    args = parser.parse_args()

    model, tokenizer, device = load_model(config.OUTPUT_DIR)

    if args.text:
        print(f"EN: {args.text} -> Q: {translate(args.text, model, tokenizer, device)}")
    else:
        print("Type q to quit.")
        while True:
            text = input("Type a word or sentence to translate: ")
            if text.lower() == 'q': break
            print(f"Translation: {translate(text, model, tokenizer, device)}")
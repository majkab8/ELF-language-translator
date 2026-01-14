import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import config
import argparse


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

def translate(text, model, tokenizer, device):
    input_text = config.PREFIX + text
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs.input_ids,
        max_length=64,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Text to translate")
    args = parser.parse_args()

    model, tokenizer, device = load_model(config.OUTPUT_DIR)

    if args.text:
        print(f"EN: {args.text} -> Q: {translate(args.text, model, tokenizer, device)}")
    else:
        print("Wpisz 'q' aby wyjść.")
        while True:
            text = input("Podaj zdanie po angielsku: ")
            if text.lower() == 'q': break
            print(f"Tłumaczenie: {translate(text, model, tokenizer, device)}")
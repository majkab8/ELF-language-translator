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
    input_text = config.MODEL_PREFIX + text
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs.input_ids,
        max_length=40,
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
        print("\n" + "=" * 30)
        print("FAST MODEL TEST:")
        words_to_test = ["star", "sword", "king", "love", "river"]

        for word in words_to_test:
            translation = translate(word, model, tokenizer, device)
            print(f"EN: {word:10} -> Q: {translation}")
        print("=" * 30 + "\n")

        print("Press q to quit.")
        while True:
            text = input("Type a word or sentence to translate: ")
            if text.lower() == 'q': break
            print(f"Translation: {translate(text, model, tokenizer, device)}")
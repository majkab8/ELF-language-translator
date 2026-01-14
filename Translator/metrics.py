import evaluate
import numpy as np

chrf = evaluate.load("chrf")
cer = evaluate.load("cer")

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = {}
    result["chrf"] = chrf.compute(predictions=decoded_preds, references=decoded_labels)["score"]
    result["cer"] = cer.compute(predictions=decoded_preds, references=decoded_labels)

    return result
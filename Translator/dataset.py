from torch.utils.data import Dataset
import config

class ElvishDataset(Dataset):
    def __init__(self, csv_file, tokenizer, prefix=""):
        self.df = csv_file
        self.tokenizer = tokenizer
        self.prefix = prefix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        src = self.prefix + str(row["english"])
        tgt = str(row["elvish"])

        max_len = getattr(config, "MAX_LENGTH", 128)

        inputs = self.tokenizer(src, return_tensors="pt",
                                 truncation=True, max_length=max_len)
        labels = self.tokenizer(tgt, return_tensors="pt",
                                 truncation=True, max_length=max_len)

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze()
        }
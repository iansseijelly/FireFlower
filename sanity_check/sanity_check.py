

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AdamW
from transformers import BertTokenizer, BertForMaskedLM


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="./cache")
model = BertForMaskedLM.from_pretrained("bert-base-uncased", cache_dir="./cache")


print(tokenizer.mask_token_id)


data_path = "./sanity_check/fire_flower.txt"

with open(data_path, "r") as f:
    data = f.read().split("\n")

for sentence in data:
    if len(sentence) < 50:
        data.remove(sentence)


# tokenize input
inputs = tokenizer(
    data,
    max_length=512,
    truncation=True,
    padding=True,
    return_tensors="pt",
    )

print(inputs.keys())


# prepare masks
inputs["labels"] = inputs["input_ids"].detach().clone()


random_tensor = torch.rand(inputs["input_ids"].shape)

# filter out CLS, SEP and PAD tokens
masks = (random_tensor < 0.15) * (inputs["input_ids"] != 101) * (inputs["input_ids"] != 102) * (inputs["input_ids"] != 0)

# apply masks
non_zero_indices = []
for i in range(inputs["input_ids"].shape[0]):
    non_zero_indices.extend(torch.flatten(masks[i].nonzero()).tolist())

for i in range(inputs["input_ids"].shape[0]):
    inputs["input_ids"][i, non_zero_indices] = 103


class BookDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, index):
        input_ids = self.encodings["input_ids"][index]
        attention_mask = self.encodings["attention_mask"][index]
        token_type_ids = self.encodings["token_type_ids"][index]
        labels = self.encodings["labels"][index]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
    
dataset = BookDataset(inputs)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

epochs = 20
optimizer = AdamW(model.parameters(), lr=1e-5)

model.train()

for epoch in range(epochs):
    loop = tqdm(dataloader)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())
    





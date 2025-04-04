
import torch
from transformers import BertForMaskedLM, BertTokenizer, TrainingArguments
from torch.utils.data import DataLoader

from tacit_learn.tokenizer import Tokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("example_input.txt", "r") as f:
    example_input = f.read()

tokenizer = Tokenizer()
model = BertForMaskedLM.from_pretrained("google-bert/bert-base-uncased", cache_dir="cache").to(device)

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

encoded_input = tokenizer(example_input, return_tensors="pt")

encoded_input["input_ids"] = encoded_input["input_ids"][:, :512].to(device)
encoded_input["token_type_ids"] = encoded_input["token_type_ids"][:, :512].to(device)
encoded_input["attention_mask"] = encoded_input["attention_mask"][:, :512].to(device)

logits = model(labels=encoded_input["input_ids"], **encoded_input)


train_dataloader = DataLoader([encoded_input], shuffle=True, batch_size=8)
eval_dataloader = DataLoader([encoded_input], batch_size=8)


from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)


from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

import torch
from accelerate.test_utils.testing import get_backend

device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
model.to(device)


from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    outputs = model(labels=encoded_input["input_ids"], **encoded_input)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)

    print(f"Epoch {epoch} loss: {loss.item()}")


# test

model.eval()
outputs = model(labels=encoded_input["input_ids"], **encoded_input)
loss = outputs.loss
print(loss)

# run a fill test

model.eval()
outputs = model(labels=encoded_input["input_ids"], **encoded_input)
# retrieve index of [MASK]
mask_token_index = (encoded_input["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

predicted_token_id = outputs.logits[0, mask_token_index].argmax(axis=-1)
predicted_word = tokenizer.decode(predicted_token_id)
print(f"Predicted word: {predicted_word}")



test_input = "START INST addi RD x1 RS1 x0 [MASK] 0 TIMESTAMP 0 END"


labels = tokenizer(test_input, return_tensors="pt")["input_ids"]
print(labels)

# Decode the tokens
for i in range(len(labels[0])):
    token = tokenizer.decode(labels[0][i])
    print(token, end=" ")
    if token == "END":
        print()

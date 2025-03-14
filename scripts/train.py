from transformers import AutoTokenizer, BertForMaskedLM, BertTokenizer
import torch

# Load pretrained tokenizer and model
pretrained_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir="cache")
model = BertForMaskedLM.from_pretrained("google-bert/bert-base-uncased", cache_dir="cache")


input_str = "C0:      [MASK] [1] pc=[00000000800007b0] W[r10=000000000000001b][1] R[r 2=0000000080030bf0] R[r 0=0000000000000000] inst=[00013503] ld      a0, 0(sp)"
input_ref = "C0:      16413 [1] pc=[00000000800007b0] W[r10=000000000000001b][1] R[r 2=0000000080030bf0] R[r 0=0000000000000000] inst=[00013503] ld      a0, 0(sp)"






# Load the tokenizer from the saved files (this will load all the configurations and "weights")
tokenizer = BertTokenizer(vocab_file="vocab.txt", model_max_length=512)

# Now use the properly initialized tokenizer
inputs = tokenizer(input_str, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
predicted_word = tokenizer.decode(predicted_token_id)
print(f"Predicted word: {predicted_word}")

# For loss calculation
labels = tokenizer(input_ref, return_tensors="pt")["input_ids"]

from transformers import AutoTokenizer, BertForMaskedLM, BertTokenizer

from tacit_learn.tokenizer import Preprocessor


# Load pretrained tokenizer and model
pretrained_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir="cache")
model = BertForMaskedLM.from_pretrained("google-bert/bert-base-uncased", cache_dir="cache")





# Load the tokenizer from the saved files (this will load all the configurations and "weights")
tokenizer = BertTokenizer(vocab_file="vocab/riscv_vocab.txt", model_max_length=512)

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

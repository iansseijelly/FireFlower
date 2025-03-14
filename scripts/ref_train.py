import torch
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from datasets import load_dataset


# DistilBERT model is 67 M
model_checkpoint = "distilbert-base-uncased"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir="cache")

# load the pre-trained model
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint, cache_dir="cache")

# load the IMDB dataset
imdb_dataset = load_dataset("imdb")



text = "This is a great [MASK]."


inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits

# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]

# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")




sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))
# >>> sample
# Dataset({
#     features: ['text', 'label'],
#     num_rows: 3
# })
#
#  sample[0]
# {'text': 'There ... all.', 'label': 1}


def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# Use batched=True to activate fast multithreading!
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
# >>> tokenized_datasets
# DatasetDict({
#     train: Dataset({
#         features: ['input_ids', 'attention_mask', 'word_ids'],
#         num_rows: 25000
#     })
#     test: Dataset({
#         features: ['input_ids', 'attention_mask', 'word_ids'],
#         num_rows: 25000
#     })
#     unsupervised: Dataset({
#         features: ['input_ids', 'attention_mask', 'word_ids'],
#         num_rows: 50000
#     })
# })
# 
# >>> tokenized_datasets["test"]
# Dataset({
#     features: ['input_ids', 'attention_mask', 'word_ids'],
#     num_rows: 25000
# })
# 
# >>> tokenized_datasets["test"][0]
# {
#   'input_ids': [101, 1045, 2293, 16596, 1011, 10882, 1998, 2572, 5627, 2000, 2404, 2039, 2007, 1037, 2843, 1012, 16596, 1011, 10882, 5691, 1013, 2694, 2024, 2788, 2104, 11263, 25848, 1010, 2104, 1011, 12315, 1998, 28947, 1012, 1045, 2699, 2000, 2066, 2023, 1010, 1045, 2428, 2106, 1010, 2021, 2009, 2003, 2000, 2204, 2694, 16596, 1011, 10882, 2004, 17690, 1019, 2003, 2000, 2732, 10313, 1006, 1996, 2434, 1007, 1012, 10021, 4013, 3367, 20086, 2015, 1010, 10036, 19747, 4520, 1010, 25931, 3064, 22580, 1010, 1039, 2290, 2008, 2987, 1005, 1056, 2674, 1996, 4281, 1010, 1998, 16267, 2028, 1011, 8789, 3494, 3685, 2022, 9462, 2007, 1037, 1005, 16596, 1011, 10882, 1005, 4292, 1012, 1006, 1045, 1005, 1049, 2469, 2045, 2024, 2216, 1997, 2017, 2041, 2045, 2040, 2228, 17690, 1019, 2003, 2204, 16596, 1011, 10882, 2694, 1012, 2009, 1005, 1055, 2025, 1012, 2009, 1005, 1055, 18856, 17322, 2094, 1998, 4895, 7076, 8197, 4892, 1012, 1007, 2096, 2149, 7193, 2453, 2066, 7603, 1998, 2839, 2458, 1010, 16596, 1011, 10882, 2003, 1037, 6907, 2008, 2515, 2025, 2202, 2993, 5667, 1006, 12935, 1012, 2732, 10313, 1007, 1012, 2009, 2089, 7438, 2590, 3314, 1010, 2664, 2025, 2004, 1037, 3809, 4695, 1012, 2009, 1005, 1055, 2428, 3697, 2000, 2729, 2055, 1996, 3494, 2182, 2004, 2027, 2024, 2025, 3432, 13219, 1010, 2074, 4394, 1037, 12125, 1997, 2166, 1012, 2037, 4506, 1998, 9597, 2024, 4799, 1998, 21425, 1010, 2411, 9145, 2000, 3422, 1012, 1996, 11153, 1997, 3011, 2113, 2009, 1005, 1055, 29132, 2004, 2027, 2031, 2000, 2467, 2360, 1000, 4962, 8473, 4181, 9766, 1005, 1055, 3011, 1012, 1012, 1012, 1000, 4728, 2111, 2052, 2025, 3613, 3666, 1012, 8473, 4181, 9766, 1005, 1055, 11289, 2442, 2022, 3810, 1999, 2037, 8753, 2004, 2023, 10634, 1010, 10036, 1010, 9996, 5493, 1006, 3666, 2009, 2302, 4748, 16874, 7807, 2428, 7545, 2023, 2188, 1007, 19817, 6784, 4726, 19817, 19736, 3372, 1997, 1037, 2265, 13891, 2015, 2046, 2686, 1012, 27594, 2121, 1012, 2061, 1010, 3102, 2125, 1037, 2364, 2839, 1012, 1998, 2059, 3288, 2032, 2067, 2004, 2178, 3364, 1012, 15333, 4402, 2480, 999, 5759, 2035, 2058, 2153, 1012, 102],
#   'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#   'word_ids': [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 23, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 63, 63, 63, 64, 65, 66, 67, 68, 69, 69, 70, 71, 72, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 130, 130, 131, 132, 132, 132, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 233, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 248, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 270, 271, 272, 273, 274, 275, 276, 277, 277, 277, 278, 278, 278, 279, 280, 281, 282, 282, 283, 284, 285, 286, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 305, 305, 306, 307, 308, 309, 310, 311, None]
# }







chunk_size = 128


# Slicing produces a list of lists for each feature
tokenized_samples = tokenized_datasets["train"][:3]

for idx, sample in enumerate(tokenized_samples["input_ids"]):
    print(f"'>>> Review {idx} length: {len(sample)}'")




concatenated_examples = {
    k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
}
total_length = len(concatenated_examples["input_ids"])
print(f"'>>> Concatenated reviews length: {total_length}'")



chunks = {
    k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
    for k, t in concatenated_examples.items()
}

for chunk in chunks["input_ids"]:
    print(f"'>>> Chunk length: {len(chunk)}'")




def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(group_texts, batched=True)


tokenizer.decode(lm_datasets["train"][1]["input_ids"])
tokenizer.decode(lm_datasets["train"][1]["labels"])



from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

samples = [lm_datasets["train"][i] for i in range(2)]
for sample in samples:
    _ = sample.pop("word_ids")

for chunk in data_collator(samples)["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")





import collections
import numpy as np

from transformers import default_data_collator

wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)



train_size = 10_000
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
downsampled_dataset


from transformers import TrainingArguments

batch_size = 64
# Show the training loss with every epoch
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    fp16=True,
    logging_steps=logging_steps,
)



from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

import math

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.train()

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
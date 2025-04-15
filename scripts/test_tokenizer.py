from tacit_learn.tokenizer import Tokenizer


with open("./data/rocket-hello.canonicalized.out", "r") as f:
    example_input = f.read()


example_input = example_input[:1500]

# print(example_input)

# Load the tokenizer from the saved files (this will load all the configurations and "weights")
tokenizer = Tokenizer()

# Tokenize the input
tokens = tokenizer(example_input, return_tensors="pt", max_length=200)

# Print the tokens
# print(tokens)
print(tokens["input_ids"])

# Decode the tokens
for i in range(len(tokens["input_ids"][0])):
    token = tokenizer.decode(tokens["input_ids"][0][i])
    print(token, end=" ")


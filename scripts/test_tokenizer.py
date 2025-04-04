from tacit_learn.tokenizer import Tokenizer


with open("example_input.txt", "r") as f:
    example_input = f.read()

# print(example_input)

# Load the tokenizer from the saved files (this will load all the configurations and "weights")
tokenizer = Tokenizer()

# Tokenize the input
tokens = tokenizer(example_input, return_tensors="pt")

# Print the tokens
print(tokens)

# Decode the tokens
for i in range(len(tokens["input_ids"][0])):
    token = tokenizer.decode(tokens["input_ids"][0][i])
    print(token, end=" ")
    if token == "END":
        print()

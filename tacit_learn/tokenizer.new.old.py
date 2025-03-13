import torch


class Tokenizer:
    def __init__(self):
        pass

    def __call__(self, trace: str, return_tensors: str = "pt") -> torch.Tensor:
        return torch.randint(0, 60, (len(trace), 60), dtype=torch.float32)

    def decode(self, tensor: torch.Tensor) -> str:
        return "".join([chr(int(t)) for t in tensor.argmax(dim=1)])
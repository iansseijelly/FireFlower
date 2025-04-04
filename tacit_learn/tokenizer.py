import re
import struct

import torch
from transformers import BertTokenizer

class Tokenizer(BertTokenizer):

    # meta tokens
    META_TOKENS = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "START",
        "END",
        "INST",
        "RS1",
        "RS2",
        "RD",
        "CSR",
        "FUNCT3",
        "FUNCT7",
        "IMM",
        "TIMESTAMP",
    ]

    # riscv vocabs
    RISC_V_VOCABS = [
        # registers
        "x0",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x16",
        "x17",
        "x18",
        "x19",
        "x20",
        "x21",
        "x22",
        "x23",
        "x24",
        "x25",
        "x26",
        "x27",
        "x28",
        "x29",
        "x30",
        "x31",
        "ra",
        "sp",
        "gp",
        "tp",
        "t0",
        "t1",
        "t2",
        "s0",
        "fp",
        "s1",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "s8",
        "s9",
        "s10",
        "s11",
        "t3",
        "t4",
        "t5",
        "t6",

        # instructions
        "unimp",
        "ebreak",
        "sbreak",
        "ret",
        "jr",
        "jalr",
        "j",
        "jal",
        "call",
        "tail",
        "jump",
        "nop",
        "lui",
        "li",
        "mv",
        "move",
        "zext",
        "andi",
        "and",
        "beqz",
        "beq",
        "blez",
        "bgez",
        "bge",
        "bgeu",
        "ble",
        "bleu",
        "bltz",
        "bgtz",
        "blt",
        "bltu",
        "bgt",
        "bgtu",
        "bnez",
        "bne",
        "addi",
        "add",
        "la",
        "lla",
        "lga",
        "la",
        "neg",
        "slli",
        "sll",
        "srli",
        "srl",
        "srai",
        "sra",
        "sub",
        "lb",
        "lbu",
        "lh",
        "lhu",
        "lw",
        "not",
        "ori",
        "or",
        "lpad",
        "auipc",
        "seqz",
        "snez",
        "sltz",
        "sgtz",
        "slti",
        "slt",
        "sltiu",
        "sltu",
        "sgt",
        "sgtu",
        "sb",
        "sh",
        "sw",
        "fence",
        "rdcycle",
        "rdinstret",
        "rdtime",
        "rdcycleh",
        "rdinstreth",
        "rdtimeh",
        "ecall",
        "scall",
        "xori",
        "xor",
        "lwu",
        "ld",
        "sd",
        "sext",
        "addiw",
        "addw",
        "negw",
        "slliw",
        "sllw",
        "srliw",
        "srlw",
        "sraiw",
        "sraw",
        "subw",
        "mul",
        "mulh",
        "mulhu",
        "mulhsu",
        "div",
        "divu",
        "rem",
        "remu",
        "mulw",
        "divw",
        "divuw",
        "remw",
        "remuw",
        "csrrw",
        "csrrs",
        "csrrc",
        "csrrwi",
        "csrrsi",
        "csrrci",
        "c.unimp",
        "c.ebreak",
        "c.jr",
        "c.jalr",
        "c.j",
        "c.jal",
        "c.beqz",
        "c.bnez",
        "c.lwsp",
        "c.lw",
        "c.swsp",
        "c.sw",
        "c.nop",
        "c.nop",
        "c.mv",
        "c.lui",
        "c.li",
        "c.addi4spn",
        "c.addi16sp",
        "c.addi",
        "c.add",
        "c.sub",
        "c.and",
        "c.or",
        "c.xor",
        "c.slli",
        "c.srli",
        "c.srai",
        "c.slli64",
        "c.srli64",
        "c.srai64",
        "c.andi",
        "c.addiw",
        "c.addw",
        "c.subw",
        "c.ldsp",
        "c.ld",
        "c.sdsp",
        "c.sd",
        "c.fldsp",
        "c.fld",
        "c.fsdsp",
        "c.fsd",
        "c.flwsp",
        "c.flw",
        "c.fswsp",
        "c.fsw",
    ]

    def __init__(self):
        self._generate_vocab()
        
        super().__init__(
            vocab_file="vocab/riscv_vocab.txt",
            do_lower_case=False,
            do_basic_tokenize=True,
            never_split=Tokenizer.RISC_V_VOCABS,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=False,
        )
        pass

    def _generate_vocab(self):
        vocab_content = []

        # meta tokens
        for token in Tokenizer.META_TOKENS:
            vocab_content.append(token)

        # riscv vocabs
        for vocab in Tokenizer.RISC_V_VOCABS:
            vocab_content.append(vocab)

        # numbers
        vocab_content.append("-")
        for i in range(10000):
            vocab_content.append(str(i))

        num_lines = len(vocab_content)

        # total_tokens = 30523
        self.total_tokens = num_lines

        # Calculate how many unused tokens to add
        unused_lines = self.total_tokens - num_lines
        print(f"Current vocabulary has {num_lines} tokens. Adding {unused_lines} unused tokens.")

        for i in range(unused_lines):
            vocab_content.append(f"[unused{i}]")

        # Then append the unused tokens
        with open("vocab/riscv_vocab.txt", "w") as f:
            for token in vocab_content:
                f.write(f"{token}\n")
        
        print(f"Updated vocabulary file now has {self.total_tokens} tokens.")
    
    @property
    def num_tokens(self) -> int:
        return self.total_tokens

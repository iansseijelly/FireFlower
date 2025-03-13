import re
import struct

import torch


class Disassembler:

    COMPRESSED_MASK = 0x3

    OPCODE_POS = 0
    RD_POS = 7
    FUNC3_POS = 12
    RS1_POS = 15
    RS2_POS = 20
    FUNC7_POS = 25

    OPCODE_MASK = 0x7F << OPCODE_POS
    RD_MASK = 0x1F << RD_POS
    FUNC3_MASK = 0x07 << FUNC3_POS
    RS1_MASK = 0x1F << RS1_POS
    RS2_MASK = 0x1F << RS2_POS
    FUNC7_MASK = 0x7F << FUNC7_POS

    OPCODE_MAP = {
        0b0110011: "R",
        0b0000011: "I",
        0b0010011: "I",
        0b0001111: "I",
        0b0010111: "U",
        0b0110111: "U",
        0b1100011: "SB",
        0b0100011: "S",
        0b1101111: "UJ",
        0b1110011: "I",
        0b1100111: "I",
    }

    @staticmethod
    def disassemble(inst: int) -> list[int]:
        """
        Disassemble the instruction.
        """
        
        is_compressed = (inst & Disassembler.COMPRESSED_MASK) != 0b11
        if is_compressed:
            return None, None, None, None, None, None, None
        
        opcode = inst & Disassembler.OPCODE_MASK
        opcode_type = Disassembler.OPCODE_MAP[opcode]
        match (opcode_type):
            case "R":
                func3 = (inst & Disassembler.FUNC3_MASK) >> Disassembler.FUNC3_POS
                func7 = (inst & Disassembler.FUNC7_MASK) >> Disassembler.FUNC7_POS
                rd = (inst & Disassembler.RD_MASK) >> Disassembler.RD_POS
                rs1 = (inst & Disassembler.RS1_MASK) >> Disassembler.RS1_POS
                rs2 = (inst & Disassembler.RS2_MASK) >> Disassembler.RS2_POS
                return opcode, func3, func7, rd, rs1, rs2, None
            case "I":
                func3 = (inst & Disassembler.FUNC3_MASK) >> Disassembler.FUNC3_POS
                rd = (inst & Disassembler.RD_MASK) >> Disassembler.RD_POS
                rs1 = (inst & Disassembler.RS1_MASK) >> Disassembler.RS1_POS
                imm = Disassembler.i_get_imm(inst)
                return opcode, func3, None, rd, rs1, None, imm
            case "U":
                rd = (inst & Disassembler.RD_MASK) >> Disassembler.RD_POS
                imm = Disassembler.u_get_imm(inst)
                return opcode, None, None, rd, None, None, imm
            case "SB":
                func3 = (inst & Disassembler.FUNC3_MASK) >> Disassembler.FUNC3_POS
                rs1 = (inst & Disassembler.RS1_MASK) >> Disassembler.RS1_POS
                rs2 = (inst & Disassembler.RS2_MASK) >> Disassembler.RS2_POS
                imm = Disassembler.sb_get_imm(inst)
                return opcode, func3, None, None, rs1, rs2, imm
            case "UJ":
                rd = (inst & Disassembler.RD_MASK) >> Disassembler.RD_POS
                imm = Disassembler.uj_get_imm(inst)
                return opcode, None, None, None, rd, None, imm
            case "S":
                func3 = (inst & Disassembler.FUNC3_MASK) >> Disassembler.FUNC3_POS
                rs1 = (inst & Disassembler.RS1_MASK) >> Disassembler.RS1_POS
                rs2 = (inst & Disassembler.RS2_MASK) >> Disassembler.RS2_POS
                imm = Disassembler.s_get_imm(inst)
                return opcode, func3, None, None, rs1, rs2, imm

    @staticmethod
    def i_get_imm(inst: int) -> int:
        uimm = (inst & (0xFFF << 20)) >> 20
        if uimm & (1 << 11):
            uimm |= 0xFFFFF000
            uimm = struct.unpack("<i", struct.pack("<I", uimm))[0]
        return uimm

    @staticmethod
    def u_get_imm(inst: int) -> int:
        return (inst & (0xFFFFF << 12)) >> 12

    @staticmethod
    def sb_get_imm(inst: int) -> int:
        uimm = ((inst & (0xF << 8)) >> 8 << 1) + ((inst & (0x3F << 25)) >> 25 << 5) + \
            ((inst & (0x1 << 7)) >> 7 << 11) + ((inst & (0x1 << 31)) >> 31 << 12)
        if uimm & (1 << 12):
            uimm |= 0xFFFFE000
            uimm = struct.unpack("<i", struct.pack("<I", uimm))[0]
        return uimm

    @staticmethod
    def uj_get_imm(inst: int) -> int:
        uimm = ((inst & 0x3FF << 21) >> 21 << 1) + ((inst & (0x1 << 20)) >> 20 << 11) + \
            ((inst & (0xFF << 12)) >> 12 << 12) + ((inst & (0x1 << 31)) >> 31 << 20)
        if uimm & (1 << 20):
            uimm |= 0xFFE00000
            uimm = struct.unpack("<i", struct.pack("<I", uimm))[0]
        return uimm

    @staticmethod
    def s_get_imm(inst: int) -> int:
        uimm = ((inst & (0x1F << 7)) >> 7) + ((inst & (0x7F << 25)) >> 25 << 5)
        if uimm & (1 << 11):
            uimm |= 0xFFFFF000
            uimm = struct.unpack("<i", struct.pack("<I", uimm))[0]
        return uimm


class Tokenizer:
    """
    A tokenizer for the Spike trace.
    """

    METATOKEN = {
        "[S]": 0,   # start of instruction
        "[E]": 1,   # end of instruction
        "TIMESTAMP": 2,   # timestamp
        "OP": 3,    # opcode
        "FUNCT3": 4,  # funct3
        "FUNCT7": 5,  # funct7
        "RS1": 6,   # rs1
        "RS2": 7,   # rs2
        "RD": 8,    # rd
        "IMM": 9,   # immediate
    }

    def __init__(self):
        pass

    def __call__(self, trace: str, return_tensors: str = "pt") -> torch.Tensor:
        return self.tokenize(trace, return_tensors)

    def trace_to_vec(self, trace: str) -> torch.Tensor:
        """
        Convert the trace to a vector.
        """
        lines = trace.split("\n")

        tokens = []

        for line in lines:
            # split the line at whitespace, treat consecutive whitespace as a single delimiter
            fields = re.split(r'(?<!\[r)\s+', line)

            if not fields or fields[0] != "C0:":
                continue

            c0, timestamp, valid, pc, rd, rs1, rs2, inst, *inst_disassemble = fields

            # valid = valid == "[1]"

            timestamp = int(timestamp)

            # pc = pc.replace("pc=[", "").replace("]", "")
            # pc = int(pc, 16)

            inst = inst.replace("inst=[", "").replace("]", "")
            inst = int(inst, 16)

            opcode, func3, func7, rd, rs1, rs2, imm = Disassembler.disassemble(inst)


            t = [0.] * 60
            t[Tokenizer.METATOKEN["[S]"]] = 1.0
            tokens.append(t)

            t = [0.] * 60
            t[Tokenizer.METATOKEN["TIMESTAMP"]] = 1.0
            tokens.append(t)
            t = [0.] * 60
            t[len(Tokenizer.METATOKEN)] = timestamp
            tokens.append(t)
            
            t = [0.] * 60
            t[Tokenizer.METATOKEN["OP"]] = 1.0
            tokens.append(t)
            t = [0.] * 60
            for i in range(7):
                t[len(Tokenizer.METATOKEN) + 1 + i] = (opcode >> i) & 0b1
            tokens.append(t)

            if func3 is not None:
                t = [0.] * 60
                t[Tokenizer.METATOKEN["FUNCT3"]] = 1.0
                tokens.append(t)
                t = [0.] * 60
                t[len(Tokenizer.METATOKEN) + 1 + i] = 1.0
                tokens.append(t)

            if func7 is not None:
                t = [0.] * 60
                t[Tokenizer.METATOKEN["FUNCT7"]] = 1.0
                tokens.append(t)
                t = [0.] * 60
                t[len(Tokenizer.METATOKEN) + 1 + i] = 1.0
                tokens.append(t)

            if rs1 is not None:
                t = [0.] * 60
                t[Tokenizer.METATOKEN["RS1"]] = 1.0
                tokens.append(t)
                t = [0.] * 60
                t[len(Tokenizer.METATOKEN) + 1 + 7 + rs1] = 1.0
                tokens.append(t)
            
            if rs2 is not None:
                t = [0.] * 60
                t[Tokenizer.METATOKEN["RS2"]] = 1.0
                tokens.append(t)
                t = [0.] * 60
                t[len(Tokenizer.METATOKEN) + 1 + 7 + rs2] = 1.0
                tokens.append(t)
            
            if rd is not None:
                t = [0.] * 60
                t[Tokenizer.METATOKEN["RD"]] = 1.0
                tokens.append(t)
                t = [0.] * 60
                t[len(Tokenizer.METATOKEN) + 1 + 7 + rd] = 1.0
                tokens.append(t)
            
            if imm is not None:
                t = [0.] * 60
                t[Tokenizer.METATOKEN["IMM"]] = 1.0
                tokens.append(t)
                t = [0.] * 60
                t[len(Tokenizer.METATOKEN) + 1 + 7 + 32] = imm
                tokens.append(t)
            
            t = [0.] * 60
            t[Tokenizer.METATOKEN["[E]"]] = 1.0
            tokens.append(t)

        return torch.tensor(tokens, dtype=torch.float32)
    
    def tokenize(self, trace: str, return_tensors: str = "pt") -> torch.Tensor:
        vec = self.trace_to_vec(trace)
        

        
    def _disassemble_inst(self, inst: int) -> list[int]:
        """
        Disassemble the instruction.
        """
        opcode = inst & 0x7f
        return opcode

    def detokenize(self, tokens: torch.Tensor) -> str:
        """
        Detokenize the tokens.
        """

        for token in tokens:
            if token[Tokenizer.METATOKEN["[S]"]] == 1.0:
                print("[S]", end=" ")
            elif token[Tokenizer.METATOKEN["[E]"]] == 1.0:
                print("[E]")
            elif token[Tokenizer.METATOKEN["TIMESTAMP"]] == 1.0:
                print("TIMESTAMP", end="")
            elif token[Tokenizer.METATOKEN["OP"]] == 1.0:
                print("OP", end="")
            elif token[Tokenizer.METATOKEN["FUNCT3"]] == 1.0:
                print("FUNCT3", end="")
            elif token[Tokenizer.METATOKEN["FUNCT7"]] == 1.0:
                print("FUNCT7", end="")
            elif token[Tokenizer.METATOKEN["RS1"]] == 1.0:
                print("RS1", end="")
            elif token[Tokenizer.METATOKEN["RS2"]] == 1.0:
                print("RS2", end="")
            elif token[Tokenizer.METATOKEN["RD"]] == 1.0:
                print("RD", end="")
            elif token[Tokenizer.METATOKEN["IMM"]] == 1.0:
                print("IMM", end="")
            else:
                print("<val>", end=" ")
        
        return ""
        
        




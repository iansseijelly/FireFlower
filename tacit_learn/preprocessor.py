import re
import struct


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


class Preprocessor:
    """
    A preprocessor for the Spike trace.
    """
    
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

    meta_tokens = {
        "START": "START",
        "END": "END",
        "TIMESTAMP": "TIMESTAMP",
        "OPCODE": "OPCODE",
        "RS1": "RS1",
        "RS2": "RS2",
        "RD": "RD",
        "FUNCT3": "FUNCT3",
        "FUNCT7": "FUNCT7",
        "IMM": "IMM",
    }

    def __init__(self):
        pass

    def encode(self, trace: str) -> str:
        """
        Encode the Spike trace into string for tokenizer.
        """
        lines = trace.split("\n")

        result = ""

        prev_timestamp = None

        for line in lines:
            # split the line at whitespace, treat consecutive whitespace as a single delimiter
            fields = re.split(r'(?<!\[r)\s+', line)

            if not fields or fields[0] != "C0:":
                continue

            c0, timestamp, valid, pc, rd, rs1, rs2, inst, *inst_disassemble = fields

            # valid = valid == "[1]"

            timestamp = int(timestamp)
            if prev_timestamp is not None:
                delta_time = timestamp - prev_timestamp
            else:
                delta_time = 0
            
            prev_timestamp = timestamp

            # pc = pc.replace("pc=[", "").replace("]", "")
            # pc = int(pc, 16)

            inst = inst.replace("inst=[", "").replace("]", "")
            inst = int(inst, 16)

            opcode, funct3, funct7, rd, rs1, rs2, imm = Disassembler.disassemble(inst)

            if opcode is None:
                print(f"Warning: Invalid instruction: {inst}")
                continue

            result += f"{Preprocessor.meta_tokens['START']}"
            result += f" {Preprocessor.meta_tokens['TIMESTAMP']} {delta_time}"
            result += f" {Preprocessor.meta_tokens['OPCODE']} {opcode}"
            
            if funct3 is not None:
                result += f" {Preprocessor.meta_tokens['FUNCT3']} {funct3}"
            if funct7 is not None:
                result += f" {Preprocessor.meta_tokens['FUNCT7']} {funct7}"
            if rd is not None:
                result += f" {Preprocessor.meta_tokens['RD']} {rd}"
            if rs1 is not None:
                result += f" {Preprocessor.meta_tokens['RS1']} {rs1}"
            if rs2 is not None:
                result += f" {Preprocessor.meta_tokens['RS2']} {rs2}"
            if imm is not None:
                if imm > 999:
                    imm = 999
                    print(f"Warning: IMM is too large: {imm}")
                if imm < -999:
                    imm = -999
                    print(f"Warning: IMM is too small: {imm}")
                result += f" {Preprocessor.meta_tokens['IMM']} {imm}"
            result += f" {Preprocessor.meta_tokens['END']} "

        return result

    def print_encoded(self, encoded: str):
        encoded = encoded.replace(Preprocessor.meta_tokens['END'], f"{Preprocessor.meta_tokens['END']}\n")
        print(encoded)

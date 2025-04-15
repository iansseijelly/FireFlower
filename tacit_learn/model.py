import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class FireFlowerConfig:
    n_inst: int # number of instances
    d_inst: int # instance embedding dimension
    n_reg: int # number of regions
    d_reg: int # region embedding dimension
    d_imm: int # immediate dimension
    d_bb: int # basic block embedding dimension
    d_model: int # model dimension
    n_head: int # number of attention heads
    n_layers: int # number of layers
    n_pos: int # position embedding dimension
    d_pos: int # position embedding dimension

class FireFlowerPredictor(nn.Module):
    def __init__(self, config: FireFlowerConfig):
        super().__init__()

        self.config = config
        
        self.inst_embeddings = nn.Embedding(config.n_inst, config.d_inst)
        self.reg_embeddings = nn.Embedding(config.n_reg, config.d_reg)
        self.imm_linear = nn.Linear(1, config.d_imm)

        d_in = config.d_inst + 3 * config.d_reg + config.d_imm
        self.instruction_fusion = nn.Linear(d_in, config.d_model)
        
        self.bb_linear = nn.Linear(1, config.d_bb)
        # Define fusion layer to combine instruction and basic block info
        self.fusion = nn.Linear(config.d_model + config.d_bb, config.d_model)
        
        self.pos_embeddings = nn.Embedding(config.n_pos, config.d_pos)
        # Add a projection layer to match position embedding dimension to model dimension
        self.pos_projection = nn.Linear(config.d_pos, config.d_model)
        
        # Add a layer normalization layer after the position embedding
        self.pos_ln = nn.LayerNorm(config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_head, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        self.regression_head = nn.Linear(config.d_model, 1)

    def forward(self, instructions, positions, bbtime):
        """
        Args:
            instructions (dict): dictionary containing per-field Tensors for each instruction.
                Expected keys (for each basic block, with shape [B, L]):
                    - "inst_id": LongTensor with instruction type indices.
                    - "rs1_id": LongTensor with first register indices.
                    - "rs2_id": LongTensor with second register indices.
                    - "imm": FloatTensor with immediate values (shape [B, L, 1]).
                (If some fields are missing in an instruction, your preprocessing should supply a default token.)
            positions (Tensor): LongTensor of shape [B, L] with position indices for each instruction.
            bbtime (Tensor): FloatTensor of shape [B, 1] with the overall basic block delay.
            
        Returns:
            preds (Tensor): FloatTensor of shape [B, L, 1] with predicted per-instruction latencies.
        """
        inst_embeddings = self.inst_embeddings(instructions["inst_id"])
        rd_embeddings = self.reg_embeddings(instructions["rd_id"])
        rs1_embeddings = self.reg_embeddings(instructions["rs1_id"])
        rs2_embeddings = self.reg_embeddings(instructions["rs2_id"])
        # Convert imm to float before passing to the linear layer
        imm_embeddings = self.imm_linear(instructions["imm"])

        x = torch.cat([inst_embeddings, rs1_embeddings, rs2_embeddings, rd_embeddings, imm_embeddings], dim=-1)
        x = self.instruction_fusion(x)

        # Expand BBTIME embedding for each instruction in the block.
        bb_embeddings = self.bb_linear(bbtime)
        bb_emb_expanded = bb_embeddings.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # Concatenate the instruction representation with the BBTIME embedding.
        x = torch.cat([x, bb_emb_expanded], dim=-1)  # [B, L, d_model + d_bb]
        x = self.fusion(x)  # [B, L, d_model]

        # Add positional embedding to encode order in the basic block.
        pos_emb = self.pos_embeddings(positions)  # [B, L, d_pos]
        pos_emb = self.pos_projection(pos_emb)  # [B, L, d_model]
        x = x + pos_emb

        # Apply batch normalization to the position embedding
        x = self.pos_ln(x)

        # Prepare Transformer: PyTorch's transformer expects input as (sequence_length, batch_size, d_model)
        x = x.transpose(0, 1)  # Now shape is [L, B, d_model]
        x = self.transformer(x)  # Output shape remains [L, B, d_model]
        x = x.transpose(0, 1)    # Shape back to [B, L, d_model]

        # Regression prediction: one scalar per instruction.
        preds = self.regression_head(x)  # [B, L, 1]
        return preds

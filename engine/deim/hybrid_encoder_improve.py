import torch
import torch.nn as nn

# 1. 导入父类 (原版的 TransformerEncoderBlock)
from engine.deim.hybrid_encoder import TransformerEncoderBlock
# 2. 导入我们的自研模块
from engine.extre_module.custom_nn.mlp.MCCG import MCCG

class TransformerEncoderBlock_improve(TransformerEncoderBlock):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", pe_temperature=10000, normalize_before=False):
        # 第一步：调用父类的 __init__，让它帮我们把多头注意力、Norm层、位置编码等全部初始化好
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, pe_temperature, normalize_before)
        
        # 第二步：打扫战场，把父类初始化好的、但我们不需要的原版 FFN 线性层删除，释放内存
        del self.linear1
        del self.linear2
        del self.activation
        # self.dropout 在原版里其实就是普通的 nn.Dropout，留着或者删了用下面新的都行，为了干净我们删掉
        del self.dropout
        
        # 第三步：注入灵魂，换上带有空间感知能力的 MCCG
        self.ffn = MCCG(in_features=d_model, hidden_features=dim_feedforward, out_features=d_model)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, src, src_mask=None) -> torch.Tensor:
        # 前面部分完全照抄父类逻辑
        b, c, h, w = src.size()
        src = src.flatten(2).permute(0, 2, 1)
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c, self.pe_temperature).to(src.device)
    
        residual = src
        if self.normalize_before:    
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)   
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)  
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)  
            
        # ==============================================================
        # 魔改核心区：利用 MCCG 进行 2D 空间维度的处理
        # ==============================================================
        # 1. 序列转特征图
        src_2d = src.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
        # 2. MCCG 加工
        src_2d = self.ffn(src_2d)
        # 3. 特征图转回序列
        src = src_2d.flatten(2).permute(0, 2, 1)
        src = self.dropout_ffn(src)
        # ==============================================================

        src = residual + self.dropout2(src)     
        if not self.normalize_before: 
            src = self.norm2(src)

        return src.permute(0, 2, 1).reshape(-1, c, h, w).contiguous()
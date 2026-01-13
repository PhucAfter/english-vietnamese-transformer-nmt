import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e4)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output, attn_probs

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        q_s = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k_s = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v_s = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        context, attn_weights = self.scaled_dot_product_attention(q_s, k_s, v_s, mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x): return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_out, weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, weights

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        sa_out, sa_weights = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(sa_out))
        ca_out, ca_weights = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(ca_out))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x, sa_weights, ca_weights

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_heads=8, num_layers=3, d_ff=512, dropout=0.1):
        super().__init__()
        self.enc_emb = nn.Embedding(src_vocab_size, d_model)
        self.enc_pe = PositionalEncoding(d_model)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.dec_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.dec_pe = PositionalEncoding(d_model)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def make_src_mask(self, src):
        return (src != 0).unsqueeze(1).unsqueeze(2)
    
    def make_tgt_mask(self, tgt):
        pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        len_tgt = tgt.size(1)
        sub_mask = torch.tril(torch.ones((len_tgt, len_tgt), device=tgt.device)).bool()
        return pad_mask & sub_mask
        
    def encode(self, src, src_mask):
        x = self.dropout(self.enc_pe(self.enc_emb(src)))
        enc_weights = []
        for layer in self.enc_layers:
            x, w = layer(x, src_mask)
            enc_weights.append(w)
        return x, enc_weights
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.dropout(self.dec_pe(self.dec_emb(tgt)))
        dec_weights = []
        for layer in self.dec_layers:
            x, sa_w, ca_w = layer(x, enc_output, src_mask, tgt_mask)
            dec_weights.append((sa_w, ca_w))
        return x, dec_weights

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_output, _ = self.encode(src, src_mask)
        dec_output, _ = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.fc_out(dec_output)
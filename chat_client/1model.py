# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Learnable Positional Embedding

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device)
        return x + self.pos(positions)


# Multi-Head Cosine Attention  

class MultiHeadCosineAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        B, S, D = x.size()
        H, hd = self.num_heads, self.head_dim
        Q = self.q_lin(x).view(B, S, H, hd).transpose(1,2)
        K = self.k_lin(x).view(B, S, H, hd).transpose(1,2)
        V = self.v_lin(x).view(B, S, H, hd).transpose(1,2)

        Q_norm = F.normalize(Q, dim=-1)
        K_norm = F.normalize(K, dim=-1)
        scores = torch.matmul(Q_norm, K_norm.transpose(-2,-1))

        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V).transpose(1,2).contiguous().view(B, S, D)
        return self.out_lin(context)


# Feed Forward with GLU

class FeedForwardGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff*2)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.glu(self.linear1(x), dim=-1)
        x = self.dropout(x)
        return self.linear2(x)


# Encoder Layer

class EncoderLayerSuper(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadCosineAttention(d_model, num_heads, dropout)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.ffn = FeedForwardGLU(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x, mask=None):
        y = self.attn(x, mask)
        x = self.norm1(x + self.alpha * self.drop(y))

        conv = self.conv1d(x.transpose(1,2)).transpose(1,2)
        x = self.norm2(x + self.alpha * self.drop(conv))

        y = self.ffn(x)
        x = self.norm3(x + self.alpha * self.drop(y))
        return x

 
# SuperNssanee Encoder
 
class SuperNssaneeEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, max_len=128, num_layers=2, num_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = LearnablePositionalEmbedding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayerSuper(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        x = self.dropout(x)
        all_layer_outputs = []
        for layer in self.layers:
            x = layer(x, mask)
            all_layer_outputs.append(x)
        return all_layer_outputs
 
#  Classifier
 
class SuperNssaneeClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, max_len=128, num_layers=2, num_heads=4, d_ff=512, num_classes=6, dropout=0.1):
        super().__init__()
        self.encoder = SuperNssaneeEncoder(vocab_size, d_model, max_len, num_layers, num_heads, d_ff, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model*3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, mask=None):
        all_layer_outputs = self.encoder(x, mask)
        enc = all_layer_outputs[-1]
        cls_tokens = enc[:,0,:]
        pooled = torch.cat([cls_tokens, enc.mean(dim=1), enc.max(dim=1).values], dim=-1)  # ✅ fixed
        return self.classifier(pooled)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SimpleTransformer(nn.Module):
    def __init__(self, 
                 vocab_size=3,           # Input vocab (0, 1, <PAD>)
                 output_vocab_size=2,    # Output vocab (0, 1 only)
                 d_model=128, 
                 nhead=8, 
                 num_layers=4, 
                 max_len=512, 
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.output_vocab_size = output_vocab_size
        
        # Token embedding (includes PAD token)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection (only to original vocab, no PAD prediction)
        self.output_proj = nn.Linear(d_model, output_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        x = self.dropout(x) 
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        if input_ids.is_cuda:
            causal_mask = causal_mask.cuda()
        
        # Convert attention mask to transformer format
        if attention_mask is not None:
            # attention_mask: 1 for real tokens, 0 for padding
            # transformer expects: True for padding tokens to ignore
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None
        
        # Transformer forward
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        
        # Output projection
        logits = self.output_proj(x)
        
        return logits
    
    def generate_probabilities(self, input_ids, attention_mask=None):
        """Generate next token probabilities for the last valid position"""
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            
            # Get probabilities for the last non-padding token
            if attention_mask is not None:
                # Find last real token for each sequence
                last_token_idx = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(logits.size(0))
                last_logits = logits[batch_indices, last_token_idx]
            else:
                last_logits = logits[:, -1]  # Last token
            
            probs = F.softmax(last_logits, dim=-1)
            return probs
    
    def compute_loss(self, input_ids, attention_mask, target_ids):
        """Compute cross-entropy loss for next token prediction"""
        logits = self.forward(input_ids, attention_mask)
        
        # Get logits for last valid position of each sequence
        if attention_mask is not None:
            last_token_idx = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(logits.size(0))
            pred_logits = logits[batch_indices, last_token_idx]
        else:
            pred_logits = logits[:, -1]
        
        # Cross-entropy loss
        loss = F.cross_entropy(pred_logits, target_ids)
        return loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
import numpy as np


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = query.shape[:2]
        
        # Linear projections
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask (for autoregressive generation)
        if causal_mask:
            causal_mask_tensor = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask_tensor.to(scores.device), float('-inf'))
        
        # Apply attention mask (for padding)
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, attention_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights


class AutoregressiveTransformer(nn.Module):
    """
    Standard GPT-style decoder-only transformer for sequence modeling.
    
    Architecture based on fsm_transformer_plan.md:
    - Embedding dim: 64
    - Hidden dim: 128  
    - Heads: 4
    - Layers: 2-4
    - Vocabulary: {0, 1, PAD}
    """
    
    def __init__(
        self,
        vocab_size: int = 3,  # {0, 1, PAD}
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 128,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 2
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        x = self.dropout(token_embeds + pos_embeds)
        
        # Process attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        
        # Transformer blocks
        hidden_states = []
        all_attention_weights = []
        
        for block in self.blocks:
            x, attn_weights = block(x, attention_mask)
            hidden_states.append(x)
            all_attention_weights.append(attn_weights)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        outputs = {
            'logits': logits,
            'last_hidden_state': x
        }
        
        if return_hidden_states:
            outputs['hidden_states'] = hidden_states
            outputs['attention_weights'] = all_attention_weights
            
        return outputs
    
    def extract_representations(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer: int = -1
    ) -> torch.Tensor:
        """
        Extract hidden representations at specified layer for causal state analysis.
        
        Args:
            input_ids: Input token sequences
            attention_mask: Attention mask for padding
            layer: Which layer to extract (-1 for final layer)
            
        Returns:
            Hidden states [batch_size, seq_len, embed_dim]
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, return_hidden_states=True)
            
            if layer == -1:
                return outputs['last_hidden_state']
            else:
                return outputs['hidden_states'][layer]
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """Generate sequences autoregressively."""
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
            
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        
        for _ in range(max_length):
            # Forward pass
            outputs = self.forward(generated)
            logits = outputs['logits']
            
            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < top_k_logits[:, -1:]] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences hit pad token
            if (next_token == pad_token_id).all():
                break
                
        return generated
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute autoregressive language modeling loss."""
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs['logits']
        
        # For autoregressive training, we predict the next token
        # So we use the last position's logits to predict target_ids
        last_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        
        loss = F.cross_entropy(last_logits, target_ids)
        return loss
    
    def get_next_token_probabilities(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get probability distribution over next tokens."""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs['logits'][:, -1, :]  # Last position
            probs = F.softmax(logits, dim=-1)
            return probs
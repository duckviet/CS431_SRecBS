#model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)


class SimplifiedTCELayer(nn.Module):
    def __init__(
        self,
        num_items,
        num_base_tables,
        base_table_size,
        embed_dim,
        device=None,
        padding_idx=0,
    ):
        super().__init__()
        self.num_items = num_items
        self.num_base_tables = num_base_tables
        self.base_table_size = base_table_size
        self.embed_dim = embed_dim
        self.device = device
        self.padding_idx = padding_idx

        if num_items > (base_table_size ** num_base_tables):
            print(
                f"Warning: Base table configuration might not guarantee unique embeddings."
                f" {base_table_size}^{num_base_tables} < {num_items}"
            )

        # Create N base embedding tables
        self.base_embedding_tables = nn.ModuleList(
            [
                nn.Embedding(
                    base_table_size,
                    embed_dim,
                    padding_idx=self.padding_idx,
                )
                for _ in range(num_base_tables)
            ]
        )

        # Simple learnable weights for fusing base embeddings
        self.fusion_weights = nn.Parameter(torch.randn(num_base_tables, 1))

        # Initialize weights
        for table in self.base_embedding_tables:
            nn.init.normal_(table.weight, std=0.01)
            if self.padding_idx is not None:
                with torch.no_grad():
                    table.weight[self.padding_idx].fill_(0)
        nn.init.xavier_uniform_(self.fusion_weights)

    def forward(self, item_seq):
        # Handle both 1D and 2D input shapes
        if len(item_seq.shape) == 1:
            item_seq = item_seq.unsqueeze(0)  # Add batch dimension: (seq_len,) -> (1, seq_len)
            is_1d_input = True
        else:
            is_1d_input = False

        batch_size, seq_len = item_seq.shape
        current_device = self.device or item_seq.device

        # Create non-padding mask
        non_padding_mask = (item_seq != self.padding_idx).unsqueeze(-1).float()  # (batch_size, seq_len, 1)

        # Precompute indices for all base tables using vectorized operations
        quotient = item_seq.long()  # Start with the original item IDs
        indices_list = []
        base_table_size_tensor = torch.tensor(self.base_table_size, device=current_device, dtype=torch.long)

        for _ in range(self.num_base_tables):
            indices = torch.remainder(quotient, base_table_size_tensor)
            quotient = torch.div(quotient, base_table_size_tensor, rounding_mode='floor')
            # Handle padding indices
            if self.padding_idx is not None:
                indices = torch.where(
                    item_seq == self.padding_idx,
                    torch.tensor(self.padding_idx, device=current_device, dtype=torch.long),
                    indices
                )
            indices_list.append(indices)

        # Stack indices for all base tables: (num_base_tables, batch_size, seq_len)
        indices_stack = torch.stack(indices_list, dim=0)

        # Initialize output tensor for composed embeddings
        composed_embeddings = torch.zeros(
            batch_size, seq_len, self.embed_dim, device=current_device
        )

        # Apply softmax to fusion weights once
        normalized_weights = F.softmax(self.fusion_weights, dim=0)  # (num_base_tables, 1)

        # Accumulate embeddings from all base tables without intermediate stacking
        for n in range(self.num_base_tables):
            # Get indices for current base table: (batch_size, seq_len)
            indices_n = indices_stack[n]
            # Get embeddings: (batch_size, seq_len, embed_dim)
            base_emb = self.base_embedding_tables[n](indices_n)
            # Weight for current table: scalar value
            weight_n = normalized_weights[n, 0]
            # Accumulate weighted embeddings
            composed_embeddings += weight_n * base_emb

        # Apply mask to ensure padding positions have zero vectors
        composed_embeddings = composed_embeddings * non_padding_mask

        # If input was 1D, remove the added batch dimension
        if is_1d_input:
            composed_embeddings = composed_embeddings.squeeze(0)  # (1, seq_len, embed_dim) -> (seq_len, embed_dim)

        return composed_embeddings
  
# --- New SASRec model using TCE Layer ---
class SASRec_TCE(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec_TCE, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # === Use SimplifiedTCELayer for items ===
        self.item_emb = SimplifiedTCELayer(
            num_items=self.item_num + 1,  # +1 for padding index 0
            num_base_tables=args.num_base_tables,
            base_table_size=args.base_table_size,
            embed_dim=args.hidden_units,
            device=self.dev,
            padding_idx=0  # Explicitly set padding index
        )
        # ========================================

        # Positional embedding remains standard
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        # === Use TCE Layer ===
        # Input shape: (batch_size, seq_len)
        # Output shape: (batch_size, seq_len, embed_dim)
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # ====================

        # Scaling (optional, common practice with embeddings)
        seqs *= self.item_emb.embed_dim ** 0.5  # Use embed_dim from TCE layer

        # --- Positional embedding and subsequent layers remain the same ---
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # Ensure positions for padding items (log_seqs == 0) are zeroed out
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.dev)
        )

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)  # (U, T, C)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.log2feats(log_seqs)  # (batch_size, seq_len, embed_dim)

        # === Use TCE Layer for pos/neg items ===
        # Input shape: (batch_size, seq_len)
        # Output shape: (batch_size, seq_len, embed_dim)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        # ======================================

        # Calculate logits using element-wise multiplication and sum
        # log_feats: (U, T, C), pos_embs: (U, T, C) -> pos_logits: (U, T)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # (batch_size, seq_len, embed_dim)

        # Use the embedding of the last item's output state
        final_feat = log_feats[:, -1, :]  # (batch_size, embed_dim)

        # === Use TCE Layer for candidate items ===
        # Input shape: (batch_size, num_candidates) -> e.g., (1, 101)
        # Output shape: (batch_size, num_candidates, embed_dim) -> e.g., (1, 101, 50)
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        # ========================================

        # Calculate logits by matrix multiplication
        # item_embs: (U, I, C), final_feat: (U, C) -> (U, C, 1)
        # logits: (U, I, C) @ (U, C, 1) -> (U, I, 1) -> (U, I)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits

class StochasticMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False,
                 add_zero_attn=False, kdim=None, vdim=None, stochastic_dropout=0.1,
                 stochastic_head_prob=1.0):
        super(StochasticMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.stochastic_dropout = stochastic_dropout  # Dropout rate for attention scores
        self.stochastic_head_prob = stochastic_head_prob  # Probability of keeping each head
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

        # Dropout layer for attention scores
        self.attn_dropout = nn.Dropout(stochastic_dropout)

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None):
        # Shape of query: (tgt_len, batch_size, embed_dim)
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        # Get number of heads to use based on stochastic_head_prob
        if self.training and self.stochastic_head_prob < 1.0:
            num_active_heads = max(1, int(self.num_heads * self.stochastic_head_prob))
            active_heads = torch.randperm(self.num_heads)[:num_active_heads].tolist()
        else:
            num_active_heads = self.num_heads
            active_heads = list(range(self.num_heads))

        if self._qkv_same_embed_dim:
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        else:
            q = F.linear(query, self.q_proj_weight, self.in_proj_bias[:embed_dim] if self.in_proj_bias is not None else None)
            k = F.linear(key, self.k_proj_weight, self.in_proj_bias[embed_dim:2*embed_dim] if self.in_proj_bias is not None else None)
            v = F.linear(value, self.v_proj_weight, self.in_proj_bias[2*embed_dim:] if self.in_proj_bias is not None else None)

        # Reshape for multi-head
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # Select active heads if stochastic head selection is enabled
        if num_active_heads < self.num_heads:
            q_active = torch.cat([q[i*bsz:(i+1)*bsz] for i in active_heads], dim=0)
            k_active = torch.cat([k[i*bsz:(i+1)*bsz] for i in active_heads], dim=0)
            v_active = torch.cat([v[i*bsz:(i+1)*bsz] for i in active_heads], dim=0)
        else:
            q_active = q
            k_active = k
            v_active = v

        src_len = k.size(1)
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q_active, k_active.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * num_active_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, num_active_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_weights = attn_weights.view(bsz * num_active_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        # Apply stochastic dropout to attention weights
        if self.training:
            attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.bmm(attn_weights, v_active)
        assert list(attn_output.size()) == [bsz * num_active_heads, tgt_len, self.head_dim]

        # Reshape back to original dimensions
        if num_active_heads < self.num_heads:
            # If not all heads are active, we need to pad the output with zeros for inactive heads
            full_attn_output = torch.zeros(bsz * self.num_heads, tgt_len, self.head_dim, device=attn_output.device)
            for idx, head in enumerate(active_heads):
                full_attn_output[head*bsz:(head+1)*bsz] = attn_output[idx*bsz:(idx+1)*bsz]
            attn_output = full_attn_output
        else:
            attn_output = attn_output

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            attn_weights = attn_weights.view(bsz, num_active_heads, tgt_len, src_len)
            if num_active_heads < self.num_heads:
                full_attn_weights = torch.zeros(bsz, self.num_heads, tgt_len, src_len, device=attn_weights.device)
                for idx, head in enumerate(active_heads):
                    full_attn_weights[:, head, :, :] = attn_weights[:, idx, :, :]
                attn_weights = full_attn_weights
            return attn_output, attn_weights
        else:
            return attn_output, None

class S3Rec_TCE(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(S3Rec_TCE, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # === Use SimplifiedTCELayer for items ===
        self.item_emb = SimplifiedTCELayer(
            num_items=self.item_num + 1,  # +1 for padding index 0
            num_base_tables=args.num_base_tables,
            base_table_size=args.base_table_size,
            embed_dim=args.hidden_units,
            device=self.dev,
            padding_idx=0  # Explicitly set padding index
        )
        # ========================================

        # Positional embedding remains standard
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            # Thay thế MultiheadAttention bằng StochasticMultiheadAttention
            new_attn_layer = StochasticMultiheadAttention(
                embed_dim=args.hidden_units,
                num_heads=args.num_heads,
                dropout=args.dropout_rate,
                stochastic_dropout=args.stochastic_dropout if hasattr(args, 'stochastic_dropout') else 0.1,
                stochastic_head_prob=args.stochastic_head_prob if hasattr(args, 'stochastic_head_prob') else 0.8
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        # === Use TCE Layer ===
        # Input shape: (batch_size, seq_len)
        # Output shape: (batch_size, seq_len, embed_dim)
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # ====================

        # Scaling (optional, common practice with embeddings)
        seqs *= self.item_emb.embed_dim ** 0.5  # Use embed_dim from TCE layer

        # --- Positional embedding and subsequent layers remain the same ---
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # Ensure positions for padding items (log_seqs == 0) are zeroed out
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.dev)
        )

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)  # (U, T, C)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.log2feats(log_seqs)  # (batch_size, seq_len, embed_dim)

        # === Use TCE Layer for pos/neg items ===
        # Input shape: (batch_size, seq_len)
        # Output shape: (batch_size, seq_len, embed_dim)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        # ======================================

        # Calculate logits using element-wise multiplication and sum
        # log_feats: (U, T, C), pos_embs: (U, T, C) -> pos_logits: (U, T)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # (batch_size, seq_len, embed_dim)

        # Use the embedding of the last item's output state
        final_feat = log_feats[:, -1, :]  # (batch_size, embed_dim)

        # === Use TCE Layer for candidate items ===
        # Input shape: (batch_size, num_candidates) -> e.g., (1, 101)
        # Output shape: (batch_size, num_candidates, embed_dim) -> e.g., (1, 101, 50)
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        # ========================================

        # Calculate logits by matrix multiplication
        # item_embs: (U, I, C), final_feat: (U, C) -> (U, C, 1)
        # logits: (U, I, C) @ (U, C, 1) -> (U, I, 1) -> (U, I)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
    

class S3Rec_SMA(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(S3Rec_SMA, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # Use standard embedding layer for items
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)

        # Positional embedding remains standard
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            # Use StochasticMultiheadAttention
            new_attn_layer = StochasticMultiheadAttention(
                embed_dim=args.hidden_units,
                num_heads=args.num_heads,
                dropout=args.dropout_rate,
                stochastic_dropout=args.stochastic_dropout,
                stochastic_head_prob=args.stochastic_head_prob
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        # Use standard embedding layer
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5

        # Positional embedding and subsequent layers
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        # Use standard embedding for pos/neg items
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]

        # Use standard embedding for candidate items
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
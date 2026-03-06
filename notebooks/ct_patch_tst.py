        

class ChannelTSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, 
                activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Position-wise Feed-Forward
        self.ff_channel = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, 
                shape: Tuple[int, int, int, int]=None):
        """
        src: tensor [bs x q_len x d_model] or [bs*n_vars, num_patch, d_model] or [B*M, N, P]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)

        src = self._channel_mixing(src, prev, shape=shape)

    def _channel_mixing(self, src: Tensor, 
                        prev:Optional[Tensor]=None, shape=shape):
        """
        src: tensor [bs x q_len x d_model] or [bs*n_vars, num_patch, d_model] or [B*M, N, P]

        Need to convert to: [bs*num_patch, n_vars, d_model]

        Return [B*M, N, P]
        """
        # reshape
        bs, n_vars, num_patch, d_model = shape        
        src = torch.reshape(
            src, (bs, n_vars, num_patch, d_model)
        )
        src = src.transpose(1,2)    
        
        ## Multi-Head attention
        ## Channel Mixing
        if self.res_attention:
            prev_2 = torch.reshape(prev_2, (bs, n_vars, num_patch, d_model))
            prev_2 = prev_2.transpose(1,2)
            src2, attn, scores = self.self_attn(src, src, src, prev_2)
        else:
            src2, attn = self.self_attn(src, src, src)            
        if self.store_attn:
            self.attn = attn        

        

        

        
        
        


    def _patch_tst_forward(self, src:Tensor, prev:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]

        tensor [bs*n_vars, num_patch, d_model] or [B*M, N, P]
        """        
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)            
        if self.store_attn:
            self.attn = attn
            
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




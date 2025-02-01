from typing import Optional, Tuple, Union
from diffusers import UVit2DModel

#----------------------------------------------------------------------------
# Adaptation of HuggingFace's UViTModel to use with the 
# simpleDiffusion pipeline

class UViT(UVit2DModel):
    def __init__(
        self,
        # global config
        hidden_size: int = 1024,
        use_bias: bool = False,
        hidden_dropout: float = 0.0,
        # conditioning dimensions
        cond_embed_dim: int = 768,
        micro_cond_encode_dim: int = 256,
        micro_cond_embed_dim: int = 1280,
        encoder_hidden_size: int = 768,
        # num tokens
        vocab_size: int = 8256,  # codebook_size + 1 (for the mask token) rounded
        codebook_size: int = 8192,
        # `UVit2DConvEmbed`
        in_channels: int = 768,
        block_out_channels: int = 768,
        num_res_blocks: int = 3,
        downsample: bool = False,
        upsample: bool = False,
        block_num_heads: int = 12,
        # `TransformerLayer`
        num_hidden_layers: int = 22,
        num_attention_heads: int = 16,
        # `Attention`
        attention_dropout: float = 0.0,
        # `FeedForward`
        intermediate_size: int = 2816,
        # `Norm`
        layer_norm_eps: float = 1e-6,
        ln_elementwise_affine: bool = True,
        sample_size: int = 64,
    ):
        super().__init__(
            hidden_size=hidden_size,
            use_bias=use_bias,
            hidden_dropout=hidden_dropout,
            cond_embed_dim=cond_embed_dim,
            micro_cond_encode_dim=micro_cond_encode_dim,
            micro_cond_embed_dim=micro_cond_embed_dim,
            encoder_hidden_size=encoder_hidden_size,
            vocab_size=vocab_size,
            codebook_size=codebook_size,
            in_channels=in_channels,
            block_out_channels=block_out_channels,
            num_res_blocks=num_res_blocks,
            downsample=downsample,
            upsample=upsample,
            block_num_heads=block_num_heads,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            intermediate_size=intermediate_size,
            layer_norm_eps=layer_norm_eps,
            ln_elementwise_affine=ln_elementwise_affine,
            sample_size=sample_size,
        )

    def forward(self, x, noise_labels):
        x = super().forward(input_ids=x, encoder_hidden_states=None, pooled_text_emb=None, micro_conds=noise_labels)
        return x
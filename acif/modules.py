import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Conv1dSubsampler(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, kernel_sizes: List[int] = [5, 5]):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels if i == 0 else mid_channels // 2,
                    out_channels=mid_channels if i < len(kernel_sizes) - 1 else out_channels * 2,
                    kernel_size=k,
                    stride=2,
                    padding=k // 2,
                )
            )

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):
        # x shape: (B, T, C) -> Conv1d expects (B, C, T)
        x = x.transpose(1, 2)
        
        for conv in self.conv_layers:
            x = conv(x)
            x = F.glu(x, dim=1) 
            
            # Dynamically recalculate sequence lengths based on stride and padding
            x_lens = (x_lens + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
            
        # Back to (B, T, C)
        x = x.transpose(1, 2)
        return x, x_lens

class ACIFEncoder(nn.Module):
    def __init__(self, in_channels: int, transformer_dim: int = 1024, num_layers: int = 6):
        super().__init__()
        
        # 1. Two-layer Convolutional Subsampler (4x Temporal Reduction)
        self.subsampler = Conv1dSubsampler(
            in_channels=in_channels,
            mid_channels=transformer_dim,
            out_channels=transformer_dim,
            kernel_sizes=[5, 5]
        )
        
        # 2. Six Standard Transformer Layers 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=8, 
            dim_feedforward=transformer_dim * 4, 
            dropout=0.1, 
            activation="relu", 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, x_lens):
        x, x_lens = self.subsampler(x, x_lens)
        
        B, T, _ = x.shape
        padding_mask = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T) >= x_lens.unsqueeze(1)
        
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        return x, x_lens

class ContinuousIntegrateAndFire(nn.Module):
    def __init__(self, in_dim: int = 1024, out_dim: int = 4096):
        super().__init__()
        
        # 1. Alpha Predictor (Determines token boundaries)
        self.alpha_proj = nn.Linear(in_dim, 1)
        
        # 2. FFN
        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, out_dim)
        )

    def forward(self, features):
        # Predict alphas and squeeze to (B, T)
        alphas = torch.sigmoid(self.alpha_proj(features)).squeeze(-1)
        
        # Expand features to the LLM embedding dimension (B, T, 4096)
        frame_proj = self.feature_proj(features)
        
        return frame_proj, alphas

    @torch.no_grad()
    def infer_integrate(self, frame_proj: torch.Tensor, alphas: torch.Tensor, threshold: float = 1.0):
        B, T_x, D = frame_proj.shape
        integrated_embeds = []
        
        for b in range(B):
            b_embeds = []
            accum_feat = torch.zeros(D, device=frame_proj.device, dtype=frame_proj.dtype)
            accum_alpha = 0.0
            
            for t in range(T_x):
                alpha = alphas[b, t].item()
                feat = frame_proj[b, t]
                
                if accum_alpha + alpha < threshold:
                    accum_alpha += alpha
                    accum_feat += alpha * feat
                else:
                    rem = threshold - accum_alpha
                    accum_feat += rem * feat
                    b_embeds.append(accum_feat)
                    
                    # Carry over remainder to next firing window
                    surplus = alpha - rem
                    accum_alpha = surplus
                    accum_feat = surplus * feat
                    
            if accum_alpha >= 0.5:
                scale_factor = threshold / accum_alpha
                b_embeds.append(accum_feat * scale_factor)
                    
            if len(b_embeds) > 0:
                integrated_embeds.append(torch.stack(b_embeds))
            else:
                integrated_embeds.append(frame_proj[b, :1])
                
        return integrated_embeds
  
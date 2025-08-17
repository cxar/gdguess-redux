from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """MixStyle: feature-wise style mixing for domain adaptation.
    
    Based on "Domain Generalization with MixStyle" (Zhou et al., 2021).
    Mixes channel-wise statistics (mean, std) between samples in a batch.
    """
    
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        """
        Args:
            p: Probability of applying MixStyle
            alpha: Beta distribution parameter for mixing weight
            eps: Small value for numerical stability
        """
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
    
    def forward(self, x: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        """Apply MixStyle to input features.
        
        Args:
            x: Input tensor of shape (B, C, ...) where B=batch, C=channels
            seed: Random seed for reproducibility
            
        Returns:
            Mixed features with same shape as input
        """
        if not self.training or torch.rand(1).item() > self.p:
            return x
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        B = x.shape[0]
        if B < 2:
            return x  # Need at least 2 samples to mix
        
        # Compute statistics along spatial dimensions (keep batch and channel dims)
        # For 2D: (B, C, H, W) -> compute over (H, W)
        # For 1D: (B, C, T) -> compute over (T)
        dims = list(range(2, x.dim()))
        
        mu = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        
        # Normalize features
        x_normalized = (x - mu) / sig
        
        # Generate random permutation for mixing
        perm_idx = torch.randperm(B, device=x.device)
        
        # Sample mixing weights from Beta distribution
        lam = torch.from_numpy(np.random.beta(self.alpha, self.alpha, size=(B, 1))).float().to(x.device)
        
        # Expand lambda to match feature dimensions
        for _ in range(x.dim() - 2):
            lam = lam.unsqueeze(-1)
        
        # Mix statistics
        mu_mixed = lam * mu + (1 - lam) * mu[perm_idx]
        sig_mixed = lam * sig + (1 - lam) * sig[perm_idx]
        
        # Apply mixed statistics
        x_mixed = x_normalized * sig_mixed + mu_mixed
        
        return x_mixed


def mixstyle(feats: torch.Tensor, p: float = 0.5, alpha: float = 0.1, seed: int | None = None) -> torch.Tensor:
    """Functional interface for MixStyle augmentation.
    
    Args:
        feats: Feature tensor of shape (B, C, ...) or (C, ...)
        p: Probability of applying MixStyle
        alpha: Beta distribution parameter
        seed: Random seed
        
    Returns:
        Augmented features
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Add batch dimension if needed
    if feats.dim() < 2:
        raise ValueError("Features must have at least 2 dimensions (channels, ...)")
    
    if feats.dim() == 2:
        # Assume (C, T) -> add batch dim
        feats = feats.unsqueeze(0)
        added_batch_dim = True
    else:
        added_batch_dim = False
    
    # Apply MixStyle
    mixstyle_layer = MixStyle(p=p, alpha=alpha)
    mixstyle_layer.train()  # Ensure it's in training mode
    
    result = mixstyle_layer(feats, seed)
    
    # Remove batch dimension if we added it
    if added_batch_dim:
        result = result.squeeze(0)
    
    return result


class AdaptiveMixStyle(nn.Module):
    """Adaptive MixStyle that adjusts mixing strength based on feature statistics."""
    
    def __init__(self, p: float = 0.5, alpha_min: float = 0.05, alpha_max: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.eps = eps
    
    def forward(self, x: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        """Apply adaptive MixStyle where alpha depends on feature diversity."""
        if not self.training or torch.rand(1).item() > self.p:
            return x
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        B = x.shape[0]
        if B < 2:
            return x
        
        # Compute statistics
        dims = list(range(2, x.dim()))
        mu = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        
        # Measure feature diversity (coefficient of variation across batch)
        mu_batch = mu.mean(dim=0)  # (1, C, 1, ...)
        cv = (mu.std(dim=0) / (mu_batch.abs() + self.eps)).mean().item()
        
        # Adapt alpha based on diversity (more diverse -> less mixing)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (1 - min(cv, 1.0))
        
        # Apply MixStyle with adaptive alpha
        x_normalized = (x - mu) / sig
        perm_idx = torch.randperm(B, device=x.device)
        
        lam = torch.from_numpy(np.random.beta(alpha, alpha, size=(B, 1))).float().to(x.device)
        for _ in range(x.dim() - 2):
            lam = lam.unsqueeze(-1)
        
        mu_mixed = lam * mu + (1 - lam) * mu[perm_idx]
        sig_mixed = lam * sig + (1 - lam) * sig[perm_idx]
        
        return x_normalized * sig_mixed + mu_mixed
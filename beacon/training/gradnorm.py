"""
GradNorm: Gradient Normalization for Adaptive Loss Balancing.

Reference: Chen et al., "GradNorm: Gradient Normalization for Adaptive 
Loss Balancing in Deep Multitask Networks" (ICML 2018)

Learns task-specific loss weights that equalize gradient norms across tasks,
preventing any single loss from dominating training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class GradNorm(nn.Module):
    """
    Learnable loss weight balancing via gradient normalization.
    
    Maintains log-scale weights for each loss component and adjusts them
    so that gradient norms from each task remain balanced. Tasks that
    are learning too slowly get higher weights, and vice versa.
    
    Usage:
        gradnorm = GradNorm(
            loss_names=['profile', 'tf_identity', 'site_supervision'],
            alpha=1.5,
        )
        
        # In training loop:
        weighted_losses = gradnorm.reweight(losses, shared_params)
        total_loss = sum(weighted_losses.values())
        total_loss.backward()
        gradnorm.update(losses, shared_params, optimizer)
    """
    
    def __init__(
        self,
        loss_names: List[str],
        alpha: float = 1.5,
        lr: float = 0.025,
        initial_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            loss_names: Names of loss components to balance
            alpha: Asymmetry parameter controlling how aggressively to
                   rebalance (higher = more aggressive). 1.5 works well.
            lr: Learning rate for weight updates
            initial_weights: Optional initial weights (default: uniform)
        """
        super().__init__()
        self.loss_names = list(loss_names)
        self.n_tasks = len(loss_names)
        self.alpha = alpha
        self.lr = lr
        
        # Log-scale weights (learnable)
        if initial_weights is not None:
            init_vals = torch.tensor(
                [initial_weights.get(name, 1.0) for name in loss_names]
            ).log()
        else:
            init_vals = torch.zeros(self.n_tasks)
        
        self.log_weights = nn.Parameter(init_vals)
        
        # Track initial loss values for relative inverse training rate
        self.register_buffer(
            'initial_losses',
            torch.ones(self.n_tasks),
        )
        self.initialized = False
        
        # Weight optimizer (separate from model optimizer)
        self._weight_optimizer = None
    
    @property
    def weights(self) -> torch.Tensor:
        """Current task weights (exponentiated from log scale)."""
        return self.log_weights.exp()
    
    def get_weight_dict(self) -> Dict[str, float]:
        """Return current weights as a name->value dict."""
        w = self.weights.detach()
        return {name: w[i].item() for i, name in enumerate(self.loss_names)}
    
    def _ensure_optimizer(self):
        """Lazily create the weight optimizer."""
        if self._weight_optimizer is None:
            self._weight_optimizer = torch.optim.Adam(
                [self.log_weights], lr=self.lr
            )
    
    def reweight(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Apply learned weights to losses.
        
        Args:
            losses: Dict of loss name -> scalar tensor
            
        Returns:
            Dict of weighted losses
        """
        # Initialize baseline losses on first call
        if not self.initialized:
            for i, name in enumerate(self.loss_names):
                if name in losses:
                    val = losses[name].detach()
                    if val.item() > 1e-8:
                        self.initial_losses[i] = val
            self.initialized = True
        
        weights = self.weights
        weighted = {}
        for i, name in enumerate(self.loss_names):
            if name in losses:
                weighted[name] = weights[i] * losses[name]
            else:
                weighted[name] = torch.tensor(0.0, device=self.log_weights.device)
        
        return weighted
    
    def compute_grad_norms(
        self,
        losses: Dict[str, torch.Tensor],
        shared_layer: nn.Module,
    ) -> torch.Tensor:
        """
        Compute gradient norms of each weighted loss w.r.t. shared layer.
        
        Args:
            losses: Dict of unweighted loss tensors
            shared_layer: Shared network layer to compute gradients for
                         (typically the last shared layer of backbone)
        
        Returns:
            Tensor of gradient norms [n_tasks]
        """
        weights = self.weights
        grad_norms = torch.zeros(self.n_tasks, device=self.log_weights.device)
        
        # Get the shared layer parameters
        shared_params = list(shared_layer.parameters())
        if not shared_params:
            return grad_norms
        
        # Use the last parameter of shared layer (standard choice)
        W = shared_params[-1]
        
        for i, name in enumerate(self.loss_names):
            if name not in losses:
                continue
            loss = losses[name]
            if not loss.requires_grad or loss.item() < 1e-12:
                continue
            
            # Compute gradient of w_i * L_i w.r.t. W
            weighted_loss = weights[i] * loss
            
            grad = torch.autograd.grad(
                weighted_loss, W, 
                retain_graph=True, 
                create_graph=True,
                allow_unused=True,
            )
            
            if grad[0] is not None:
                grad_norms[i] = grad[0].norm(2)
        
        return grad_norms
    
    def update(
        self,
        losses: Dict[str, torch.Tensor],
        shared_layer: nn.Module,
    ):
        """
        Update loss weights based on GradNorm algorithm.
        
        Should be called after the main backward pass but before optimizer.step().
        
        Args:
            losses: Dict of unweighted loss tensors (must have grad)
            shared_layer: Shared layer for gradient computation
        """
        self._ensure_optimizer()
        
        with torch.enable_grad():
            # Compute gradient norms
            grad_norms = self.compute_grad_norms(losses, shared_layer)
            
            # Skip if no valid gradients
            if grad_norms.sum().item() < 1e-12:
                return
            
            # Mean gradient norm (target)
            mean_norm = grad_norms.mean().detach()
            
            # Compute inverse training rates
            current_losses = torch.stack([
                losses.get(name, torch.tensor(0.0, device=self.log_weights.device)).detach()
                for name in self.loss_names
            ])
            
            # Relative inverse training rate: L_i(t) / L_i(0)
            ratios = current_losses / (self.initial_losses + 1e-8)
            mean_ratio = ratios.mean()
            rel_inv_rate = ratios / (mean_ratio + 1e-8)
            
            # Target gradient norms
            target_norms = mean_norm * (rel_inv_rate ** self.alpha)
            
            # GradNorm loss: |G_i - target_i|_1
            gradnorm_loss = (grad_norms - target_norms.detach()).abs().sum()
            
            # Update weights
            self._weight_optimizer.zero_grad()
            gradnorm_loss.backward()
            self._weight_optimizer.step()
        
        # Renormalize weights so they sum to n_tasks
        with torch.no_grad():
            normalize_factor = self.n_tasks / self.weights.sum()
            self.log_weights.add_(normalize_factor.log())

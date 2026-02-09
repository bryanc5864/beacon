"""
PCGrad: Projecting Conflicting Gradients for Multi-Task Learning.

Reference: Yu et al., "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)

When gradients from different losses conflict (negative cosine similarity),
PCGrad projects conflicting gradients onto the normal plane of each other,
resolving the conflict while preserving non-conflicting components.
"""

import torch
from typing import Dict, List, Optional
import random


class PCGradOptimizer:
    """
    Wraps an optimizer to apply PCGrad gradient surgery.
    
    Usage:
        optimizer = AdamW(model.parameters(), lr=1e-4)
        pcgrad = PCGradOptimizer(optimizer)
        
        # In training loop:
        losses = {'profile': loss1, 'tf': loss2, 'slot': loss3}
        pcgrad.step(losses, scaler=scaler)  # handles AMP if scaler provided
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, reduction: str = 'sum'):
        """
        Args:
            optimizer: Base optimizer to wrap
            reduction: How to combine de-conflicted gradients ('sum' or 'mean')
        """
        self.optimizer = optimizer
        self.reduction = reduction
        self._params = []
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self._params.append(p)
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
    
    def step(
        self,
        losses: Dict[str, torch.Tensor],
        scaler: Optional[torch.amp.GradScaler] = None,
        grad_clip: Optional[float] = None,
    ) -> float:
        """
        Perform PCGrad update step.
        
        Args:
            losses: Dictionary of named loss tensors
            scaler: Optional GradScaler for AMP training
            grad_clip: Optional gradient clipping value
            
        Returns:
            Total gradient norm after de-conflicting
        """
        # Filter out zero/non-differentiable losses
        active_losses = {}
        for name, loss in losses.items():
            if loss.requires_grad and loss.item() > 0:
                active_losses[name] = loss
        
        if len(active_losses) == 0:
            return 0.0
        
        # If only one loss, fall back to standard gradient
        if len(active_losses) == 1:
            loss = list(active_losses.values())[0]
            self.optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self._params, grad_clip)
            
            grad_norm = self._compute_grad_norm()
            
            if scaler is not None:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()
            
            return grad_norm
        
        # Compute per-loss gradients
        task_grads = {}
        loss_names = list(active_losses.keys())

        # Get the current scale factor for manual unscaling
        inv_scale = 1.0 / scaler.get_scale() if scaler is not None else 1.0

        for name in loss_names:
            self.optimizer.zero_grad()
            loss = active_losses[name]

            if scaler is not None:
                scaler.scale(loss).backward(retain_graph=True)
            else:
                loss.backward(retain_graph=True)

            # Store gradients (manually unscaled to avoid scaler state issues)
            grads = []
            for p in self._params:
                if p.grad is not None:
                    grads.append(p.grad.detach().clone() * inv_scale)
                else:
                    grads.append(torch.zeros_like(p))
            task_grads[name] = grads

        # Apply PCGrad: project conflicting gradients
        deconflicted_grads = self._project_conflicting(task_grads, loss_names)

        # Set the de-conflicted gradients
        self.optimizer.zero_grad()
        for i, p in enumerate(self._params):
            if p.grad is None:
                p.grad = deconflicted_grads[i]
            else:
                p.grad.copy_(deconflicted_grads[i])

        if scaler is not None:
            # Re-scale gradients so scaler.unscale_() can properly unscale + do inf checks
            scale = scaler.get_scale()
            for p in self._params:
                if p.grad is not None:
                    p.grad.mul_(scale)
            scaler.unscale_(self.optimizer)

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self._params, grad_clip)

        grad_norm = self._compute_grad_norm()

        if scaler is not None:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

        return grad_norm
    
    def _project_conflicting(
        self,
        task_grads: Dict[str, List[torch.Tensor]],
        loss_names: List[str],
    ) -> List[torch.Tensor]:
        """
        Project conflicting gradients onto normal planes.
        
        For each pair of task gradients g_i, g_j:
        If cos(g_i, g_j) < 0, replace g_i with:
            g_i - (g_i . g_j / ||g_j||^2) * g_j
        """
        n_tasks = len(loss_names)
        
        # Flatten gradients for efficient computation
        flat_grads = {}
        for name in loss_names:
            flat_grads[name] = torch.cat([g.flatten() for g in task_grads[name]])
        
        # Copy for modification
        projected = {name: flat_grads[name].clone() for name in loss_names}
        
        # Random task ordering for fairness
        order = list(range(n_tasks))
        random.shuffle(order)
        
        for i_idx in order:
            name_i = loss_names[i_idx]
            for j_idx in order:
                if i_idx == j_idx:
                    continue
                name_j = loss_names[j_idx]
                
                g_i = projected[name_i]
                g_j = flat_grads[name_j]
                
                # Check for conflict
                dot = torch.dot(g_i, g_j)
                if dot < 0:
                    # Project g_i onto normal plane of g_j
                    norm_sq = torch.dot(g_j, g_j)
                    if norm_sq > 1e-12:
                        projected[name_i] = g_i - (dot / norm_sq) * g_j
        
        # Combine projected gradients
        if self.reduction == 'sum':
            combined = sum(projected.values())
        else:
            combined = sum(projected.values()) / n_tasks
        
        # Unflatten back to parameter shapes
        result = []
        offset = 0
        for p in self._params:
            numel = p.numel()
            result.append(combined[offset:offset + numel].reshape(p.shape))
            offset += numel
        
        return result
    
    def _compute_grad_norm(self) -> float:
        """Compute total gradient norm."""
        total_norm = 0.0
        for p in self._params:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

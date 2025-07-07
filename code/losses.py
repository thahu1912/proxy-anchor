import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

class Bayesian_Proxy_Anchor(torch.nn.Module):
    """
    Uncertainty-aware Bayesian Proxy Anchor Loss
    
    This implementation extends the original Proxy Anchor loss with uncertainty estimation
    by modeling both embeddings and proxies as Gaussian distributions with learnable variances.
    """
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, uncertainty_weight=0.1, 
                 min_uncertainty=1e-6, max_uncertainty=1.0):
        torch.nn.Module.__init__(self)
        
        # Proxy Anchor Initialization (mean)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
        # Proxy uncertainty initialization (variance)
        self.proxy_uncertainties = torch.nn.Parameter(torch.ones(nb_classes, sz_embed).cuda() * 0.1)
        nn.init.constant_(self.proxy_uncertainties, 0.1)
        
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.uncertainty_weight = uncertainty_weight
        self.min_uncertainty = min_uncertainty
        self.max_uncertainty = max_uncertainty
        
    def forward(self, X, X_uncertainty, T):
        """
        Forward pass for Bayesian Proxy Anchor Loss
        
        Args:
            X: Embeddings (batch_size, sz_embed)
            X_uncertainty: Embedding uncertainties (batch_size, sz_embed)
            T: Labels (batch_size,)
        """
        P = self.proxies
        P_uncertainty = torch.clamp(self.proxy_uncertainties, 
                                   min=self.min_uncertainty, 
                                   max=self.max_uncertainty)
        
        # Normalize embeddings and proxies
        X_norm = l2_norm(X)
        P_norm = l2_norm(P)
        
        # Calculate cosine similarity
        cos = F.linear(X_norm, P_norm)
        
        # Calculate uncertainty-aware similarity
        # Combine embedding and proxy uncertainties
        total_uncertainty = X_uncertainty.unsqueeze(1) + P_uncertainty.unsqueeze(0)  # (batch_size, nb_classes, sz_embed)
        
        # Uncertainty-weighted similarity
        uncertainty_weight = 1.0 / (1.0 + total_uncertainty.mean(dim=-1))  # (batch_size, nb_classes)
        uncertainty_aware_cos = cos * uncertainty_weight
        
        # Create one-hot encodings
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
        
        # Calculate exponential terms with uncertainty adjustment
        pos_exp = torch.exp(-self.alpha * (uncertainty_aware_cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (uncertainty_aware_cos + self.mrg))
        
        # Find valid proxies
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_valid_proxies = len(with_pos_proxies)
        
        # Calculate similarity sums
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        # Main loss terms
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        
        # Uncertainty regularization terms
        embedding_uncertainty_reg = torch.mean(X_uncertainty)
        proxy_uncertainty_reg = torch.mean(P_uncertainty)
        
        # Total loss
        main_loss = pos_term + neg_term
        uncertainty_reg = self.uncertainty_weight * (embedding_uncertainty_reg + proxy_uncertainty_reg)
        loss = main_loss + uncertainty_reg
        
        return loss
    
    def get_uncertainty_metrics(self, X, X_uncertainty, T):
        """
        Get uncertainty metrics for analysis
        """
        P = self.proxies
        P_uncertainty = torch.clamp(self.proxy_uncertainties, 
                                   min=self.min_uncertainty, 
                                   max=self.max_uncertainty)
        
        X_norm = l2_norm(X)
        P_norm = l2_norm(P)
        cos = F.linear(X_norm, P_norm)
        
        total_uncertainty = X_uncertainty.unsqueeze(1) + P_uncertainty.unsqueeze(0)
        uncertainty_weight = 1.0 / (1.0 + total_uncertainty.mean(dim=-1))
        
        # Calculate a simple uncertainty-aware loss
        uncertainty_loss = torch.mean(total_uncertainty) * self.uncertainty_weight
        return uncertainty_loss 
    
# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


@dataclass
class BayesianTripletConfig:
    """Configuration class for Bayesian Triplet Loss parameters."""
    margin: float = 0.3  # Increased margin for better separation
    uncertainty_weight: float = 0.05  # Reduced weight to avoid over-regularization
    min_uncertainty: float = 1e-6
    max_uncertainty: float = 1.0
    temperature: float = 1.0
    loss_scale: float = 10.0  # Scaling factor for loss magnitude
    adaptive_margin: bool = True
    uncertainty_regularization: bool = True
    triplet_mining: str = 'hardest'  # 'hardest', 'semihard', 'all'
    distance_metric: str = 'euclidean'  # 'euclidean', 'cosine', 'manhattan'
    uncertainty_type: str = 'gaussian'  # 'gaussian', 'laplace', 'uniform'


class BayesianTripletLoss(nn.Module):
    """
    Bayesian Triplet Loss with Uncertainty Estimation
    
    This implementation extends the traditional triplet loss by incorporating
    uncertainty estimation for both embeddings and the margin parameter.
    """
    
    def __init__(self, margin=0.3, uncertainty_weight=0.05, loss_scale=10.0,
                 adaptive_margin=True, triplet_mining='hardest', distance_metric='euclidean',
                 uncertainty_type='gaussian', min_uncertainty=1e-6, max_uncertainty=1.0,
                 temperature=1.0, uncertainty_regularization=True, device='cuda', **kwargs):
        super(BayesianTripletLoss, self).__init__()
        
        # Create configuration
        self.config = BayesianTripletConfig(
            margin=margin,
            uncertainty_weight=uncertainty_weight,
            loss_scale=loss_scale,
            min_uncertainty=min_uncertainty,
            max_uncertainty=max_uncertainty,
            temperature=temperature,
            adaptive_margin=adaptive_margin,
            uncertainty_regularization=uncertainty_regularization,
            triplet_mining=triplet_mining,
            distance_metric=distance_metric,
            uncertainty_type=uncertainty_type
        )
        self.device = device
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate the configuration parameters."""
        assert self.config.triplet_mining in ['hardest', 'semihard', 'all'], \
            f"triplet_mining must be one of ['hardest', 'semihard', 'all'], got {self.config.triplet_mining}"
        assert self.config.distance_metric in ['euclidean', 'cosine', 'manhattan'], \
            f"distance_metric must be one of ['euclidean', 'cosine', 'manhattan'], got {self.config.distance_metric}"
        assert self.config.uncertainty_type in ['gaussian', 'laplace', 'uniform'], \
            f"uncertainty_type must be one of ['gaussian', 'laplace', 'uniform'], got {self.config.uncertainty_type}"
        assert self.config.margin >= 0, f"margin must be non-negative, got {self.config.margin}"
        assert 0 <= self.config.uncertainty_weight <= 1, \
            f"uncertainty_weight must be between 0 and 1, got {self.config.uncertainty_weight}"
            
    def forward(self, embeddings, labels, uncertainties=None):
        """
        Forward pass of the Bayesian Triplet Loss.
        
        Args:
            embeddings: Feature embeddings of shape (batch_size, embedding_dim)
            labels: Ground truth labels of shape (batch_size,)
            uncertainties: Uncertainty estimates of shape (batch_size, embedding_dim). 
                          If None, will be initialized as small random values.
            
        Returns:
            Computed loss value
        """
        # If uncertainties are not provided, initialize them
        if uncertainties is None:
            batch_size, embedding_dim = embeddings.size()
            uncertainties = torch.rand(batch_size, embedding_dim, device=embeddings.device) * 0.1
        
        # Input validation
        self._validate_inputs(embeddings, uncertainties, labels)
        
        # Normalize uncertainties and ensure they are valid
        uncertainties = torch.clamp(uncertainties, 
                                  min=self.config.min_uncertainty,
                                  max=self.config.max_uncertainty)
        
        # Check for NaN or inf values and replace them
        uncertainties = torch.where(torch.isnan(uncertainties) | torch.isinf(uncertainties),
                                  torch.full_like(uncertainties, self.config.min_uncertainty),
                                  uncertainties)
        
        # Compute pairwise distances with uncertainty
        distances, distance_uncertainties = self._compute_uncertainty_aware_distances(
            embeddings, uncertainties
        )
        
        # Mine triplets
        triplets = self._mine_triplets(distances, labels)
        
        if len(triplets) == 0:
            return torch.zeros(1, requires_grad=True, device=self.device)
        
        # Compute triplet losses with uncertainty
        triplet_losses = []
        for anchor_idx, positive_idx, negative_idx in triplets:
            loss = self._compute_single_triplet_loss(
                distances, distance_uncertainties, 
                anchor_idx, positive_idx, negative_idx
            )
            if loss is not None:
                triplet_losses.append(loss)
        
        if len(triplet_losses) == 0:
            return torch.zeros(1, requires_grad=True, device=self.device)
        
        # Combine losses
        main_loss = torch.stack(triplet_losses).mean()
        
        # Add uncertainty regularization if enabled
        if self.config.uncertainty_regularization:
            uncertainty_reg = self._compute_uncertainty_regularization(uncertainties)
            total_loss = main_loss + self.config.uncertainty_weight * uncertainty_reg
        else:
            total_loss = main_loss
        
        # Final safety check for NaN or inf values
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.zeros(1, requires_grad=True, device=self.device)
            
        return total_loss
    
    def _validate_inputs(self, embeddings, uncertainties, labels):
        """Validate input tensors."""
        assert embeddings.size(0) == labels.size(0) == uncertainties.size(0), \
            f"Batch sizes must match: embeddings({embeddings.size(0)}), " \
            f"uncertainties({uncertainties.size(0)}), labels({labels.size(0)})"
        assert embeddings.size() == uncertainties.size(), \
            f"Embedding and uncertainty shapes must match: {embeddings.size()} vs {uncertainties.size()}"
        assert embeddings.dim() == 2, f"embeddings must be 2D, got {embeddings.dim()}D"
        assert uncertainties.dim() == 2, f"uncertainties must be 2D, got {uncertainties.dim()}D"
        assert labels.dim() == 1, f"labels must be 1D, got {labels.dim()}D"
    
    def _compute_uncertainty_aware_distances(self, embeddings, uncertainties):
        """
        Compute pairwise distances with uncertainty estimation.
        
        Returns:
            Tuple of (distances, distance_uncertainties)
        """
        batch_size = embeddings.size(0)
        
        if self.config.distance_metric == 'euclidean':
            # Compute Euclidean distances
            distances = torch.cdist(embeddings, embeddings, p=2)
            
            # Add small epsilon to avoid numerical issues
            distances = distances + 1e-8
            
            # Compute uncertainty in distances
            diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)  # (batch, batch, dim)
            distances_unsqueezed = distances.unsqueeze(-1)  # Avoid division by zero
            
            # Compute partial derivatives with numerical stability
            partial_derivs = diff / distances_unsqueezed  # (batch, batch, dim)
            
            # Propagate uncertainties
            uncertainties_unsqueezed = uncertainties.unsqueeze(1)  # (batch, 1, dim)
            distance_uncertainties = torch.sqrt(
                torch.sum(partial_derivs**2 * uncertainties_unsqueezed**2, dim=-1) + 1e-8
            )
            
        elif self.config.distance_metric == 'cosine':
            # Compute cosine distances
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            similarities = torch.matmul(embeddings_norm, embeddings_norm.t())
            distances = 1 - similarities
            
            # For cosine distance, uncertainty is more complex
            distance_uncertainties = torch.sqrt(
                torch.sum(uncertainties**2, dim=1, keepdim=True) + 
                torch.sum(uncertainties**2, dim=1, keepdim=True).t()
            ) / math.sqrt(embeddings.size(1))
            
        else:  # manhattan
            # Compute Manhattan distances
            distances = torch.cdist(embeddings, embeddings, p=1)
            
            # For Manhattan distance, uncertainty is sum of individual uncertainties
            distance_uncertainties = torch.sum(uncertainties, dim=1, keepdim=True) + \
                                   torch.sum(uncertainties, dim=1, keepdim=True).t()
        
        return distances, distance_uncertainties
    
    def _mine_triplets(self, distances, labels):
        """
        Mine triplets based on the specified strategy.
        
        Returns:
            List of (anchor_idx, positive_idx, negative_idx) tuples
        """
        batch_size = distances.size(0)
        triplets = []
        
        for anchor_idx in range(batch_size):
            anchor_label = labels[anchor_idx]
            
            # Find positive and negative indices
            positive_mask = labels == anchor_label
            negative_mask = labels != anchor_label
            
            positive_indices = torch.where(positive_mask)[0]
            negative_indices = torch.where(negative_mask)[0]
            
            # Skip if no positives or negatives
            if len(positive_indices) < 1 or len(negative_indices) < 1:
                continue
            
            # Remove self from positives
            positive_indices = positive_indices[positive_indices != anchor_idx]
            if len(positive_indices) < 1:
                continue
            
            # Get distances for this anchor
            anchor_distances = distances[anchor_idx]
            pos_distances = anchor_distances[positive_indices]
            neg_distances = anchor_distances[negative_indices]
            
            if self.config.triplet_mining == 'hardest':
                # Hardest positive and hardest negative
                hardest_positive = positive_indices[torch.argmax(pos_distances)]
                hardest_negative = negative_indices[torch.argmin(neg_distances)]
                triplets.append((anchor_idx, hardest_positive.item(), hardest_negative.item()))
                
            elif self.config.triplet_mining == 'semihard':
                # Semi-hard negative mining
                for pos_idx in positive_indices:
                    pos_dist = anchor_distances[pos_idx]
                    
                    # Find semi-hard negatives
                    semi_hard_mask = (neg_distances > pos_dist) & (neg_distances < pos_dist + self.config.margin)
                    semi_hard_indices = negative_indices[semi_hard_mask]
                    
                    if len(semi_hard_indices) > 0:
                        # Choose the hardest semi-hard negative
                        hardest_semi_hard = semi_hard_indices[torch.argmin(anchor_distances[semi_hard_indices])]
                        triplets.append((anchor_idx, pos_idx.item(), hardest_semi_hard.item()))
                    else:
                        # Fall back to hardest negative
                        hardest_negative = negative_indices[torch.argmin(neg_distances)]
                        triplets.append((anchor_idx, pos_idx.item(), hardest_negative.item()))
                        
            else:  # 'all'
                # All possible triplets
                for pos_idx in positive_indices:
                    for neg_idx in negative_indices:
                        triplets.append((anchor_idx, pos_idx.item(), neg_idx.item()))
        
        return triplets
    
    def _compute_single_triplet_loss(self, distances, distance_uncertainties,
                                   anchor_idx, positive_idx, negative_idx):
        """
        Compute loss for a single triplet with uncertainty.
        
        Returns:
            Computed triplet loss or None if invalid
        """
        # Get distances
        d_pos = distances[anchor_idx, positive_idx]
        d_neg = distances[anchor_idx, negative_idx]
        
        # Get uncertainties
        u_pos = distance_uncertainties[anchor_idx, positive_idx]
        u_neg = distance_uncertainties[anchor_idx, negative_idx]
        
        # Compute adaptive margin based on uncertainties
        if self.config.adaptive_margin:
            # Higher uncertainty leads to larger margin
            uncertainty_factor = (u_pos + u_neg) / 2.0
            adaptive_margin = self.config.margin * (1.0 + uncertainty_factor)
        else:
            adaptive_margin = self.config.margin
        
        # Compute base triplet loss
        base_loss = torch.relu(d_pos - d_neg + adaptive_margin)
        
        # Compute uncertainty-aware triplet loss
        if self.config.uncertainty_type == 'gaussian':
            # Gaussian uncertainty model - use uncertainty as confidence
            uncertainty_weight = 1.0 / (1.0 + (u_pos + u_neg) / 2.0)
            loss = base_loss * uncertainty_weight
            
        elif self.config.uncertainty_type == 'laplace':
            # Laplace uncertainty model
            uncertainty_weight = torch.exp(-(u_pos + u_neg) / 2.0)
            loss = base_loss * uncertainty_weight
            
        else:  # uniform
            # Uniform uncertainty model
            uncertainty_weight = 1.0 / (1.0 + torch.max(u_pos, u_neg))
            loss = base_loss * uncertainty_weight
        
        # Scale the loss to ensure it's in the expected range
        loss = loss * self.config.loss_scale
        
        # Apply temperature scaling
        loss = loss / self.config.temperature
        
        return loss if loss > 0 else None
    
    def _compute_uncertainty_regularization(self, uncertainties):
        """
        Compute uncertainty regularization term.
        
        Returns:
            Regularization loss
        """
        # Ensure uncertainties are within bounds and not too small
        uncertainties = torch.clamp(uncertainties, 
                                  min=self.config.min_uncertainty,
                                  max=self.config.max_uncertainty)
        
        # Encourage reasonable uncertainty levels
        mean_uncertainty = torch.mean(uncertainties)
        target_uncertainty = (self.config.min_uncertainty + self.config.max_uncertainty) / 2.0
        
        # Use a more stable regularization approach
        uncertainty_variance = torch.var(uncertainties)
        mean_deviation = torch.abs(mean_uncertainty - target_uncertainty)
        
        # Encourage diversity in uncertainties (but not too much)
        diversity_penalty = torch.relu(uncertainty_variance - 0.05)
        
        # Penalize very low uncertainties (encourage exploration)
        low_uncertainty_penalty = torch.relu(0.005 - mean_uncertainty)
        
        # Penalize very high uncertainties (encourage confidence)
        high_uncertainty_penalty = torch.relu(mean_uncertainty - 0.3)
        
        regularization = mean_deviation + 0.05 * diversity_penalty + \
                        0.05 * low_uncertainty_penalty + 0.05 * high_uncertainty_penalty
        
        return regularization
    
    def get_loss_components(self, embeddings, labels, uncertainties=None):
        """
        Get individual loss components for analysis.
        
        Returns:
            Dictionary containing loss components
        """
        # If uncertainties are not provided, initialize them
        if uncertainties is None:
            batch_size, embedding_dim = embeddings.size()
            uncertainties = torch.rand(batch_size, embedding_dim, device=embeddings.device) * 0.1
        
        # Ensure uncertainties are valid
        uncertainties_safe = torch.clamp(uncertainties, 
                                       min=self.config.min_uncertainty,
                                       max=self.config.max_uncertainty)
        
        # Compute main loss
        total_loss = self.forward(embeddings, labels, uncertainties_safe)
        
        # Compute uncertainty regularization separately
        uncertainty_reg = self._compute_uncertainty_regularization(uncertainties_safe)
        
        # Compute mean uncertainty safely
        mean_uncertainty = torch.mean(uncertainties_safe)
        
        return {
            'total_loss': total_loss,
            'uncertainty_regularization': uncertainty_reg,
            'mean_uncertainty': mean_uncertainty,
            'batch_size': embeddings.size(0)
        }
    
class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings = False)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    


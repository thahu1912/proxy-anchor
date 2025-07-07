import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses

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

class Statistical_Proxy_Anchor(torch.nn.Module):
    """
    Simple Statistical Proxy Anchor Loss - More numerically stable version
    
    This is a simplified version that avoids complex statistical computations
    that can lead to NaN values.
    """
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, stat_weight=0.01, 
                 ema_decay=0.9, stat_adjust_weight=0.15):
        torch.nn.Module.__init__(self)
        
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
        # Statistical parameters
        self.class_centers = torch.nn.Parameter(torch.zeros(nb_classes, sz_embed).cuda(), requires_grad=False)
        self.class_variances = torch.nn.Parameter(torch.ones(nb_classes, sz_embed).cuda() * 0.1, requires_grad=False)
        
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.stat_weight = stat_weight
        self.ema_decay = ema_decay  # EMA decay rate (0.9 = 90% old, 10% new)
        self.stat_adjust_weight = stat_adjust_weight  # Weight for statistical adjustment
        
    def forward(self, X, T):
        """
        Forward pass for Statistical Proxy Anchor Loss with Variance
        
        Args:
            X: Embeddings (batch_size, sz_embed)
            T: Labels (batch_size,)
        """
        P = self.proxies
        
        # Normalize embeddings and proxies
        X_norm = l2_norm(X)
        P_norm = l2_norm(P)
        
        # Basic cosine similarity
        cos = F.linear(X_norm, P_norm)  # (batch_size, nb_classes)
        
        # Statistical adjustment with variance
        stat_adjustment = torch.zeros_like(cos)
        for i in range(self.nb_classes):
            class_mask = (T == i)
            if class_mask.sum() > 0:
                # Get embeddings for this class
                class_embeddings = X_norm[class_mask]  # (num_class_samples, sz_embed)
                
                # Compute center and variance
                class_center = class_embeddings.mean(dim=0)  # (sz_embed,)
                class_var = class_embeddings.var(dim=0, unbiased=False)  # (sz_embed,)
                
                # Clamp variance for numerical stability
                class_var = torch.clamp(class_var, min=1e-6, max=10.0)
                
                # Center similarity
                proxy = P_norm[i]  # (sz_embed,)
                center_sim = F.cosine_similarity(proxy.unsqueeze(0), self.class_centers[i].unsqueeze(0))
                
                # Variance-based weight (lower variance = higher confidence)
                var_weight = 1.0 / (1.0 + class_var.mean())
                
                # Combined adjustment
                stat_adjustment[:, i] = center_sim * var_weight * self.stat_adjust_weight
        
        # Combine similarities
        combined_sim = cos + stat_adjustment
        
        # Create one-hot encodings
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
        
        # Calculate exponential terms
        pos_exp = torch.exp(-self.alpha * (combined_sim - self.mrg))
        neg_exp = torch.exp(self.alpha * (combined_sim + self.mrg))
        
        # Find valid proxies
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_valid_proxies = len(with_pos_proxies)
        
        # Calculate similarity sums
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        # Main loss terms
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        
        # Simple regularization
        center_reg = torch.mean(torch.abs(self.class_centers))
        
        # Total loss
        main_loss = pos_term + neg_term
        regularization = self.stat_weight * center_reg
        total_loss = main_loss + regularization
        
        return total_loss


    def update_centers(self, X, T):
        """
        Update class centers and track variance statistics after optimizer.step().
        Call this in your training loop: criterion.update_centers(m.detach(), y.detach())
        """
        X_norm = l2_norm(X)
        with torch.no_grad():
            for i in range(self.nb_classes):
                class_mask = (T == i)
                if class_mask.sum() > 0:
                    class_embeddings = X_norm[class_mask]
                    class_center = class_embeddings.mean(dim=0)
                    class_var = class_embeddings.var(dim=0, unbiased=False)
                    
                    # Update center with configurable EMA
                    new_center = self.ema_decay * self.class_centers[i] + (1 - self.ema_decay) * class_center
                    self.class_centers[i].copy_(new_center)
                    
                    # Update variance tracking with configurable EMA
                    new_var = self.ema_decay * self.class_variances[i] + (1 - self.ema_decay) * class_var
                    self.class_variances[i].copy_(new_var)

    
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
    
class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.loss_func = losses.NPairsLoss(l2_reg_weight=l2_reg, normalize_embeddings = False)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss

# ===== IMPROVED PROXY ANCHOR LOSSES =====

class Adaptive_Proxy_Anchor(torch.nn.Module):
    """
    Proxy Anchor with Adaptive Margin based on class difficulty
    """
    def __init__(self, nb_classes, sz_embed, base_mrg=0.1, alpha=32, adaptive_weight=0.1):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
        # Adaptive margins for each class
        self.adaptive_margins = torch.nn.Parameter(torch.ones(nb_classes).cuda() * base_mrg)
        
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.base_mrg = base_mrg
        self.alpha = alpha
        self.adaptive_weight = adaptive_weight
        
    def forward(self, X, T):
        P = self.proxies
        
        cos = F.linear(l2_norm(X), l2_norm(P))
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
        
        # Use adaptive margins for each class
        adaptive_mrg = self.base_mrg + self.adaptive_margins.unsqueeze(0)
        
        pos_exp = torch.exp(-self.alpha * (cos - adaptive_mrg))
        neg_exp = torch.exp(self.alpha * (cos + adaptive_mrg))
        
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_valid_proxies = len(with_pos_proxies)
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        
        # Regularization for adaptive margins
        margin_reg = self.adaptive_weight * torch.mean(torch.abs(self.adaptive_margins))
        
        loss = pos_term + neg_term + margin_reg
        return loss

class Curriculum_Proxy_Anchor(torch.nn.Module):
    """
    Proxy Anchor with Curriculum Learning - gradually increase difficulty
    """
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, curriculum_steps=10):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.curriculum_steps = curriculum_steps
        self.current_step = 0
        
    def set_curriculum_step(self, step):
        """Set current curriculum step (0 to curriculum_steps)"""
        self.current_step = min(step, self.curriculum_steps)
        
    def forward(self, X, T):
        P = self.proxies
        
        cos = F.linear(l2_norm(X), l2_norm(P))
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
        
        # Curriculum: gradually increase margin and alpha
        curriculum_ratio = self.current_step / self.curriculum_steps
        adaptive_mrg = self.mrg * (0.5 + 0.5 * curriculum_ratio)
        adaptive_alpha = self.alpha * (0.5 + 0.5 * curriculum_ratio)
        
        pos_exp = torch.exp(-adaptive_alpha * (cos - adaptive_mrg))
        neg_exp = torch.exp(adaptive_alpha * (cos + adaptive_mrg))
        
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_valid_proxies = len(with_pos_proxies)
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        
        loss = pos_term + neg_term
        return loss

class MultiScale_Proxy_Anchor(torch.nn.Module):
    """
    Proxy Anchor with Multi-Scale Similarity
    """
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, scales=[0.5, 1.0, 2.0]):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.scales = scales
        
    def forward(self, X, T):
        P = self.proxies
        
        # Multi-scale similarities
        multi_scale_loss = 0
        for scale in self.scales:
            X_scaled = X * scale
            P_scaled = P * scale
            
            cos = F.linear(l2_norm(X_scaled), l2_norm(P_scaled))
            P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
            N_one_hot = 1 - P_one_hot
            
            pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
            neg_exp = torch.exp(self.alpha * (cos + self.mrg))
            
            with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
            num_valid_proxies = len(with_pos_proxies)
            
            P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
            N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
            
            pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
            neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
            
            multi_scale_loss += (pos_term + neg_term) / len(self.scales)
        
        return multi_scale_loss

class Focal_Proxy_Anchor(torch.nn.Module):
    """
    Proxy Anchor with Focal Loss for handling hard examples
    """
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, gamma=2.0, focal_weight=0.25):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        
    def forward(self, X, T):
        P = self.proxies
        
        cos = F.linear(l2_norm(X), l2_norm(P))
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
        
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        
        # Focal weighting for hard examples
        pos_confidence = torch.sigmoid(cos)
        neg_confidence = 1 - pos_confidence
        
        pos_focal_weight = (1 - pos_confidence) ** self.gamma
        neg_focal_weight = (1 - neg_confidence) ** self.gamma
        
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_valid_proxies = len(with_pos_proxies)
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp * pos_focal_weight, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp * neg_focal_weight, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        
        loss = pos_term + neg_term
        return loss

class Contrastive_Proxy_Anchor(torch.nn.Module):
    """
    Proxy Anchor combined with Contrastive Learning
    """
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, contrastive_weight=0.1, temp=0.07):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.contrastive_weight = contrastive_weight
        self.temp = temp
        
    def forward(self, X, T):
        P = self.proxies
        
        # Original Proxy Anchor
        cos = F.linear(l2_norm(X), l2_norm(P))
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
        
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_valid_proxies = len(with_pos_proxies)
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        
        proxy_loss = pos_term + neg_term
        
        # Contrastive loss between samples
        X_norm = l2_norm(X)
        sample_sim = F.linear(X_norm, X_norm) / self.temp
        
        # Create mask for positive pairs (same class)
        T_expanded = T.unsqueeze(1)
        positive_mask = (T_expanded == T_expanded.T).float()
        negative_mask = 1 - positive_mask
        
        # Remove self-similarity
        positive_mask.fill_diagonal_(0)
        
        # Contrastive loss
        exp_sim = torch.exp(sample_sim)
        log_prob = sample_sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        contrastive_loss = -(positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
        contrastive_loss = contrastive_loss.mean()
        
        total_loss = proxy_loss + self.contrastive_weight * contrastive_loss
        return total_loss

class Covariance_Bayesian_Proxy_Anchor(torch.nn.Module):
    """
    Improved Bayesian Proxy Anchor with Uncertainty-Integrated Covariance Matrix
    
    This version integrates uncertainty directly into covariance matrices and uses
    Mahalanobis distance for better metric learning with simplified regularization.
    """
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, reg_weight=0.1, 
                 min_uncertainty=1e-6, max_uncertainty=1.0):
        torch.nn.Module.__init__(self)
        
        # Proxy Anchor Initialization (mean)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
        # Uncertainty-integrated covariance matrices for each class
        # Each covariance matrix includes uncertainty information
        self.class_covariances = torch.nn.Parameter(
            torch.eye(sz_embed).unsqueeze(0).repeat(nb_classes, 1, 1).cuda() * 0.1
        )
        
        # Global uncertainty scaling factor
        self.global_uncertainty = torch.nn.Parameter(torch.ones(1).cuda() * 0.1)
        
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.reg_weight = reg_weight
        self.min_uncertainty = min_uncertainty
        self.max_uncertainty = max_uncertainty
        
    def compute_uncertainty_integrated_covariance(self, X_uncertainty, P_uncertainty):
        """
        Integrate uncertainty directly into covariance matrices
        """
        # X_uncertainty: (batch_size, sz_embed)
        # P_uncertainty: (nb_classes, sz_embed)
        
        # Create uncertainty matrices
        X_unc_mat = torch.diag_embed(X_uncertainty)  # (batch_size, sz_embed, sz_embed)
        P_unc_mat = torch.diag_embed(P_uncertainty)  # (nb_classes, sz_embed, sz_embed)
        
        # Integrate with class covariances
        integrated_cov = self.class_covariances + P_unc_mat  # (nb_classes, sz_embed, sz_embed)
        
        return integrated_cov, X_unc_mat
    
    def compute_mahalanobis_similarity(self, X, P, integrated_cov):
        """
        Compute similarity using Mahalanobis distance with uncertainty integration
        """
        # X: (batch_size, sz_embed), P: (nb_classes, sz_embed)
        # integrated_cov: (nb_classes, sz_embed, sz_embed)
        
        diff = X.unsqueeze(1) - P.unsqueeze(0)  # (batch_size, nb_classes, sz_embed)
        
        # Add small diagonal term for numerical stability
        cov_stable = integrated_cov + torch.eye(self.sz_embed).cuda() * 1e-6
        
        # Compute inverse covariance
        try:
            cov_inv = torch.inverse(cov_stable)  # (nb_classes, sz_embed, sz_embed)
        except:
            # Fallback to pseudo-inverse if singular
            cov_inv = torch.pinverse(cov_stable)
        
        # Compute Mahalanobis distance
        mahal_dist = torch.sqrt(
            torch.sum(
                diff.unsqueeze(-2) @ cov_inv.unsqueeze(0) @ diff.unsqueeze(-1),
                dim=(-2, -1)
            )
        )  # (batch_size, nb_classes)
        
        # Convert distance to similarity (inverse relationship)
        mahal_sim = torch.exp(-mahal_dist * self.global_uncertainty)
        
        return mahal_sim
    
    def forward(self, X, X_uncertainty, T):
        """
        Forward pass for Uncertainty-Integrated Covariance Bayesian Proxy Anchor
        
        Args:
            X: Embeddings (batch_size, sz_embed)
            X_uncertainty: Embedding uncertainties (batch_size, sz_embed)
            T: Labels (batch_size,)
        """
        P = self.proxies
        
        # Create proxy uncertainties (learnable per class)
        P_uncertainty = torch.clamp(
            torch.ones_like(P) * self.global_uncertainty, 
            min=self.min_uncertainty, 
            max=self.max_uncertainty
        )
        
        # Compute uncertainty-integrated covariance
        integrated_cov, X_unc_mat = self.compute_uncertainty_integrated_covariance(X_uncertainty, P_uncertainty)
        
        # Compute Mahalanobis similarity
        mahal_sim = self.compute_mahalanobis_similarity(X, P, integrated_cov)
        
        # Also compute cosine similarity for combination
        X_norm = l2_norm(X)
        P_norm = l2_norm(P)
        cos_sim = F.linear(X_norm, P_norm)
        
        # Combine Mahalanobis and cosine similarities
        combined_sim = 0.6 * mahal_sim + 0.4 * cos_sim
        
        # Create one-hot encodings
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
        
        # Calculate exponential terms
        pos_exp = torch.exp(-self.alpha * (combined_sim - self.mrg))
        neg_exp = torch.exp(self.alpha * (combined_sim + self.mrg))
        
        # Find valid proxies
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_valid_proxies = len(with_pos_proxies)
        
        # Calculate similarity sums
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        # Main loss terms
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        
        # Simplified regularization term
        # Includes covariance regularization, uncertainty regularization, and consistency
        cov_reg = torch.mean(torch.abs(self.class_covariances))
        uncertainty_reg = torch.mean(X_uncertainty) + torch.mean(P_uncertainty)
        
        # Covariance consistency: encourage similar samples to have similar covariance
        consistency_loss = 0
        for i in range(self.nb_classes):
            class_mask = (T == i)
            if class_mask.sum() > 1:
                class_embeddings = X[class_mask]
                class_cov = torch.cov(class_embeddings.T)
                target_cov = self.class_covariances[i]
                consistency_loss += F.mse_loss(class_cov, target_cov)
        
        # Total loss with simplified structure
        main_loss = pos_term + neg_term
        regularization = self.reg_weight * (cov_reg + uncertainty_reg + consistency_loss)
        total_loss = main_loss + regularization
        
        return total_loss
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

class AdaptiveTripletLoss(nn.Module):
    def __init__(self, base_margin=0.1, hard_mining=True, adaptive_weight=0.1, 
                 stat_weight=0.1, ema_decay=0.9):
        super(AdaptiveTripletLoss, self).__init__()
        self.base_margin = base_margin
        self.hard_mining = hard_mining
        self.adaptive_weight = adaptive_weight
        self.stat_weight = stat_weight
        self.ema_decay = ema_decay
        
        # Statistical tracking
        self.class_means = {}
        self.class_variances = {}

    def forward(self, feats, labels):
        batch_size = feats.size(0)
        dist_mat = torch.cdist(feats, feats, p=2)
        
        # Update class statistics
        self._update_class_stats(feats, labels)
        
        losses = []
        for i in range(batch_size):
            pos_mask = labels == labels[i]
            neg_mask = labels != labels[i]
            
            if not pos_mask.any() or not neg_mask.any():
                continue
                
            # Khoảng cách với positive (loại bỏ chính nó)
            pos_dist = dist_mat[i, pos_mask]
            pos_dist = pos_dist[pos_dist > 0]  # Remove self-distance
            
            # Khoảng cách với negative
            neg_dist = dist_mat[i, neg_mask]
            
            if len(pos_dist) == 0 or len(neg_dist) == 0:
                continue
            
            # Hard mining - lấy hardest positive và hardest negative
            hardest_pos = torch.max(pos_dist)
            hardest_neg = torch.min(neg_dist)
            
            # Statistical-based adaptive margin
            stat_margin = self._compute_statistical_margin(feats[i], labels[i])
            
            # Adaptive margin (lấy ý tưởng từ CBML)
            mean_pos = torch.mean(pos_dist)
            mean_neg = torch.mean(neg_dist)
            adaptive_margin = self.base_margin + self.adaptive_weight * (mean_neg - mean_pos)
            
            # Combine margins
            final_margin = adaptive_margin + self.stat_weight * stat_margin
            
            # Tính loss
            loss = F.relu(hardest_pos - hardest_neg + final_margin)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.zeros(1, requires_grad=True).to(feats.device)
            
        return torch.mean(torch.stack(losses))
    
    def _update_class_stats(self, feats, labels):
        """Update class means and variances using EMA"""
        with torch.no_grad():
            for label in labels.unique():
                label = label.item()
                class_mask = labels == label
                class_feats = feats[class_mask]
                
                if label not in self.class_means:
                    self.class_means[label] = class_feats.mean(dim=0)
                    self.class_variances[label] = class_feats.var(dim=0, unbiased=False)
                else:
                    # EMA update
                    new_mean = class_feats.mean(dim=0)
                    new_var = class_feats.var(dim=0, unbiased=False)
                    
                    self.class_means[label] = (self.ema_decay * self.class_means[label] + 
                                             (1 - self.ema_decay) * new_mean)
                    self.class_variances[label] = (self.ema_decay * self.class_variances[label] + 
                                                  (1 - self.ema_decay) * new_var)
    
    def _compute_statistical_margin(self, feat, label):
        """Compute margin based on class statistics"""
        label = label.item()
        
        if label not in self.class_means:
            return torch.tensor(0.0).to(feat.device)
        
        class_mean = self.class_means[label]
        class_var = self.class_variances[label]
        
        # Distance to class center
        center_dist = torch.norm(feat - class_mean, p=2)
        
        # Variance-based uncertainty (higher variance = more uncertain)
        uncertainty = torch.mean(class_var)
        
        # Statistical margin: further from center + higher uncertainty = larger margin
        stat_margin = center_dist * uncertainty
        
        return stat_margin
    
    def get_class_stats(self):
        """Get current class statistics for analysis"""
        return {
            'means': self.class_means.copy(),
            'variances': self.class_variances.copy()
        }
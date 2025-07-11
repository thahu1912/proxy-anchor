import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses

def approx_log_cp(norm_kappa, emb_dim=512):
    """
    Approximate log normalization constant for von Mises-Fisher distribution
    log C_p(κ) ≈ a + b*κ + c*κ²
    """
    if emb_dim == 64:
        est = 63 - 0.03818 * norm_kappa - 0.00671 * norm_kappa**2
    elif emb_dim == 128:
        est = 127 - 0.01909 * norm_kappa - 0.003355 * norm_kappa**2
    elif emb_dim == 256:
        est = 255 - 0.009545 * norm_kappa - 0.0016775 * norm_kappa**2
    else:  # 512 and higher
        est = 868 - 0.0002662 * norm_kappa - 0.0009685 * norm_kappa ** 2
    return est

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

class Uncertainty_Aware_Proxy_Anchor(torch.nn.Module):
    """
    Proxy Anchor with variance constraints
    Incorporates the variance regularization to control the distribution of similarities
    """
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, variance_weight=0.1, hyper_weight=0.5):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.variance_weight = variance_weight
        self.hyper_weight = hyper_weight
        
    def forward(self, X, T):
        P = self.proxies

        # Normalize embeddings and proxies
        X_norm = l2_norm(X)
        P_norm = l2_norm(P)
        
        # Calculate cosine similarity
        cos = F.linear(X_norm, P_norm)
        
        # Create one-hot encodings
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        # Calculate exponential terms
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        # Find valid proxies
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_valid_proxies = len(with_pos_proxies) if len(with_pos_proxies) > 0 else 1
        
        # Calculate similarity sums
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        # Main Proxy Anchor loss terms
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        
        # Variance constraint 
        pos_var = torch.where(P_one_hot == 1, cos, torch.zeros_like(cos))
        neg_var = torch.where(N_one_hot == 1, cos, torch.zeros_like(cos))
        
    
        pos_mean = torch.sum(pos_var, dim=1, keepdim=True) / (torch.sum(P_one_hot, dim=1, keepdim=True) + 1e-8)
        neg_mean = torch.sum(neg_var, dim=1, keepdim=True) / (torch.sum(N_one_hot, dim=1, keepdim=True) + 1e-8)
        
        # Weighted mean across positive and negative similarities
        weighted_mean = self.hyper_weight * pos_mean + (1 - self.hyper_weight) * neg_mean
        
        # Variance constraint: penalize high variance in negative similarities
        neg_variance = torch.mean(torch.pow(neg_var - weighted_mean, 2))
        
        # Combine Proxy Anchor loss with variance constraint
        loss = pos_term + neg_term + self.variance_weight * neg_variance
        
        return loss


class VonMisesFisher_Proxy_Anchor(torch.nn.Module):
    """
    Proxy Anchor with von Mises-Fisher distributions
    Replaces cosine similarity with vMF log-likelihoods
    """
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, concentration_init=1.0, temperature=0.02):
        torch.nn.Module.__init__(self)
        
        # Proxy parameters - similar to original Proxy Anchor
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        # Concentration parameters for von Mises-Fisher
        self.kappa = torch.nn.Parameter(torch.ones(nb_classes) * concentration_init)

        self.temperature = temperature
        
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def vmf_log_likelihood(self, mu1, kappa1, mu2, kappa2, temperature=None):
        """
        Compute von Mises-Fisher log-likelihood between embeddings and proxies
        
        Args:
            mu1: embedding directions (batch_size, sz_embed)
            kappa1: embedding magnitudes (batch_size,)
            mu2: proxy directions (nb_classes, sz_embed)
            kappa2: proxy concentrations (nb_classes,)
        """
        if temperature is None:
            if isinstance(self.temperature, (float, int)):
                temperature = float(self.temperature)
            else:
                temperature = self.temperature.item()

        mu1_norm = F.normalize(mu1, dim=1)  # (batch_size, sz_embed)
        mu2_norm = F.normalize(mu2, dim=1)  # (nb_classes, sz_embed)

        cos_sim = F.linear(mu1_norm, mu2_norm)  # (batch_size, nb_classes)

        # von Mises-Fisher log-likelihood: κ * μ^T * x + log C(κ)
        log_likelihood = kappa2.unsqueeze(0) * cos_sim  # (batch_size, nb_classes)

        # Add normalization constant
        norm_const = approx_log_cp(kappa2, self.sz_embed)  # (nb_classes,)
        log_likelihood += norm_const.unsqueeze(0)  # (batch_size, nb_classes)
        
        # Apply temperature scaling
        log_likelihood = log_likelihood / temperature
        
        return log_likelihood

    def forward(self, X, T):
        P = self.proxies

        X_directions = F.normalize(X, dim=1)  # (batch_size, sz_embed)
        X_magnitudes = torch.norm(X, dim=1)   # (batch_size,)
        
        P_directions = F.normalize(P, dim=1)  # (nb_classes, sz_embed)
        P_concentrations = torch.clamp(self.kappa, min=0.1)  # (nb_classes,)

        vmf_similarities = self.vmf_log_likelihood(
            mu1=X_directions, 
            kappa1=X_magnitudes, 
            mu2=P_directions, 
            kappa2=P_concentrations, 
        )
        
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        # Apply margin to vMF similarities 
        pos_exp = torch.exp(-self.alpha * (vmf_similarities - self.mrg))
        neg_exp = torch.exp(self.alpha * (vmf_similarities + self.mrg))

        # Find valid proxies
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_valid_proxies = len(with_pos_proxies) if len(with_pos_proxies) > 0 else 1
        
        # Calculate similarity sums
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
    
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        
        loss = pos_term + neg_term
        
        return loss


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
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'hard')
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
    Proxy Anchor with Multi-Scale Similarity and Stochastic Scale Sampling
    """
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, scales=[0.5, 1.0, 2.0], 
                 min_scales=1, max_scales=2, dropout_prob=0.3):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.scales = scales
        self.min_scales = min_scales
        self.max_scales = max_scales
        self.dropout_prob = dropout_prob
        
    def forward(self, X, T):
        P = self.proxies
        

        num_scales_to_use = min(len(self.scales), random.randint(self.min_scales, self.max_scales))
        chosen_scales = random.sample(self.scales, num_scales_to_use)
        
        # Bernoulli dropout: randomly drop some scales with probability
        if random.random() < self.dropout_prob and len(chosen_scales) > 1:
            num_to_drop = random.randint(0, len(chosen_scales) - 1)
            if num_to_drop > 0:
                chosen_scales = random.sample(chosen_scales, len(chosen_scales) - num_to_drop)
        
        # Multi-scale similarities with stochastic sampling
        multi_scale_loss = 0
        for scale in chosen_scales:
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
            
            multi_scale_loss += (pos_term + neg_term)
        
        # Normalize by number of chosen scales
        multi_scale_loss = multi_scale_loss / len(chosen_scales) if len(chosen_scales) > 0 else 0
        
        return multi_scale_loss

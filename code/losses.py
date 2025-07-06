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
    
class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings = False)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
import numpy as np
import torch
import logging
import code.loss.losses as losses
import json
from tqdm import tqdm
import torch.nn.functional as F
import math
import vmf_sampler as vmf

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = model(J.cuda())

                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    
    return [torch.stack(A[i]) for i in range(len(A))]

def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean

def evaluate_cos(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []
    
    cos_sim = F.linear(X, X)
    # Ensure T is on the same device as cos_sim
    T_device = T.to(cos_sim.device)
    Y = T_device[cos_sim.topk(1 + K)[1][:,1:]]
    Y = Y.float().cpu()
    
    recall = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall

def evaluate_cos_Inshop(model, query_dataloader, gallery_dataloader):
    nb_classes = query_dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    query_X, query_T = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T = predict_batchwise(model, gallery_dataloader)
    
    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)
    
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 50
    Y = []
    xs = []
    
    cos_sim = F.linear(query_X, gallery_X)

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0

        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]

            thresh = torch.max(pos_sim).item()

            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1
            
        return match_counter / m
    
    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 20, 30, 40, 50]:
        r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
                
    return recall

def evaluate_cos_SOP(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 1000
    Y = []
    xs = []
    for x in X:
        if len(xs)<10000:
            xs.append(x)
        else:
            xs.append(x)            
            xs = torch.stack(xs,dim=0)
            cos_sim = F.linear(xs,X)
            # Ensure T is on the same device as cos_sim
            T_device = T.to(cos_sim.device)
            y = T_device[cos_sim.topk(1 + K)[1][:,1:]]
            Y.append(y.float().cpu())
            xs = []
            
    # Last Loop
    xs = torch.stack(xs,dim=0)
    cos_sim = F.linear(xs,X)
    # Ensure T is on the same device as cos_sim
    T_device = T.to(cos_sim.device)
    y = T_device[cos_sim.topk(1 + K)[1][:,1:]]
    Y.append(y.float().cpu())
    Y = torch.cat(Y, dim=0)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 100, 1000]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    return recall

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

def log_ppk_vmf(mu1, kappa1, mu2, kappa2, rho=0.5):
    '''
        computes the log of the Probability Product Kernel of order rho of two p-dimensional vMFs
        given their normalized means and scales
    '''
    kappa3 = torch.linalg.norm(kappa1 * mu1 + kappa2 * mu2)
    return rho * (approx_log_cp(kappa1, p) + approx_log_cp(kappa2, p)) - approx_log_cp(rho * kappa3, p)


def log_ppk_vmf_vec(mu1, kappa1, mu2, kappa2, rho=0.5, temperature=0.02, n_samples=10):
    # mu1: normalized vectors, kappa1: norms
    # mu2: normalized vectors, kappa2: norms
    if mu1.dim() == 1:
        mu1 = mu1.unsqueeze(0)
    if kappa1.dim() == 0:
        kappa1 = kappa1.unsqueeze(0).unsqueeze(1)
    if kappa2.dim() == 0:
        kappa2.unsqueeze(0)
    if kappa2.dim() == 1:
        kappa2.unsqueeze(1)
    mu1 = torch.nn.functional.normalize(mu1, dim=1)
    mu2 = torch.nn.functional.normalize(mu2, dim=1)

    # Draw samples (scales with batchsize, not proxysize).
    distr = vmf.VonMisesFisher(loc=mu1, scale=kappa1)
    # Sampling is the most time-consuming part.
    samples = distr.rsample(torch.Size([n_samples]))




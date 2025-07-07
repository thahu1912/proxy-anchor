#!/usr/bin/env python3
"""
Test script to verify the Bayesian Triplet Loss produces expected loss ranges.
"""

import torch
import torch.nn as nn
from bayesian_triplet_loss import BayesianTripletConfig, BayesianTripletLoss

def test_loss_range():
    """Test that the loss values are in the expected range."""
    
    print("Testing Bayesian Triplet Loss range...")
    
    # Test configuration
    config = BayesianTripletConfig(
        margin=0.3,
        uncertainty_weight=0.05,
        adaptive_margin=True,
        triplet_mining='hardest',
        distance_metric='euclidean',
        uncertainty_type='gaussian'
    )
    
    # Create loss function
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = BayesianTripletLoss(config, device=device)
    
    # Test data
    batch_size = 16
    embedding_dim = 128
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Config: margin={config.margin}, uncertainty_weight={config.uncertainty_weight}")
    print()
    
    # Test multiple runs to see the range
    for run in range(5):
        embeddings = torch.randn(batch_size, embedding_dim).to(device)
        uncertainties = torch.rand(batch_size, embedding_dim).to(device) * 0.1
        labels = torch.randint(0, 5, (batch_size,)).to(device)
        
        try:
            components = loss_fn.get_loss_components(embeddings, uncertainties, labels)
            
            print(f"Run {run + 1}:")
            total_loss = components['total_loss']
            uncertainty_reg = components['uncertainty_regularization']
            mean_uncertainty = components['mean_uncertainty']
            
            print(f"  Total Loss: {total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss:.4f}")
            print(f"  Uncertainty Reg: {uncertainty_reg.item() if isinstance(uncertainty_reg, torch.Tensor) else uncertainty_reg:.4f}")
            print(f"  Mean Uncertainty: {mean_uncertainty.item() if isinstance(mean_uncertainty, torch.Tensor) else mean_uncertainty:.4f}")
            
            # Check if loss is in expected range
            loss_value = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
            if 5.0 <= loss_value <= 20.0:
                print("  ✅ Loss is in expected range (5-20)")
            else:
                print(f"  ⚠️  Loss ({loss_value:.4f}) is outside expected range (5-20)")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
        
        print()
    
    # Test with different uncertainty levels
    print("Testing with different uncertainty levels...")
    
    uncertainty_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    for uncertainty_level in uncertainty_levels:
        embeddings = torch.randn(batch_size, embedding_dim).to(device)
        uncertainties = torch.rand(batch_size, embedding_dim).to(device) * uncertainty_level
        labels = torch.randint(0, 5, (batch_size,)).to(device)
        
        try:
            components = loss_fn.get_loss_components(embeddings, uncertainties, labels)
            
            print(f"Uncertainty level {uncertainty_level}:")
            total_loss = components['total_loss']
            mean_uncertainty = components['mean_uncertainty']
            
            print(f"  Total Loss: {total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss:.4f}")
            print(f"  Mean Uncertainty: {mean_uncertainty.item() if isinstance(mean_uncertainty, torch.Tensor) else mean_uncertainty:.4f}")
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
        
        print()
    
    print("Testing completed!")

if __name__ == "__main__":
    test_loss_range() 
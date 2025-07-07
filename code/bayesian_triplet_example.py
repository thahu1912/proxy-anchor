"""
Example usage of the Bayesian Triplet Loss

This script demonstrates how to integrate the Bayesian Triplet Loss
into your deep metric learning pipeline with uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from bayesian_triplet_loss import BayesianTripletConfig, BayesianTripletLoss, create_bayesian_triplet_loss


def example_basic_usage():
    """Example of basic Bayesian Triplet Loss usage."""
    
    # Method 1: Using BayesianTripletConfig
    config = BayesianTripletConfig(
        margin=0.1,
        uncertainty_weight=0.1,
        adaptive_margin=True,
        triplet_mining='hardest',
        distance_metric='euclidean',
        uncertainty_type='gaussian'
    )
    
    bayesian_loss = BayesianTripletLoss(config, device='cuda')
    
    # Method 2: Using dictionary
    config_dict = {
        'margin': 0.1,
        'uncertainty_weight': 0.1,
        'adaptive_margin': True,
        'triplet_mining': 'hardest',
        'distance_metric': 'euclidean',
        'uncertainty_type': 'gaussian'
    }
    
    bayesian_loss_2 = create_bayesian_triplet_loss(config_dict, device='cuda')
    
    return bayesian_loss, bayesian_loss_2


def example_uncertainty_estimation_model():
    """Example of a model that outputs both embeddings and uncertainties."""
    
    class UncertaintyAwareEmbeddingModel(nn.Module):
        def __init__(self, input_dim=784, embedding_dim=128):
            super(UncertaintyAwareEmbeddingModel, self).__init__()
            
            # Shared feature extractor
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # Embedding head
            self.embedding_head = nn.Sequential(
                nn.Linear(256, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
            
            # Uncertainty head
            self.uncertainty_head = nn.Sequential(
                nn.Linear(256, embedding_dim),
                nn.Softplus()  # Ensure positive uncertainties
            )
            
        def forward(self, x):
            features = self.feature_extractor(x)
            embeddings = self.embedding_head(features)
            uncertainties = self.uncertainty_head(features)
            
            return embeddings, uncertainties
    
    return UncertaintyAwareEmbeddingModel()


def example_training_loop():
    """Example training loop with Bayesian Triplet Loss."""
    
    # Create model
    model = example_uncertainty_estimation_model().cuda()
    
    # Create Bayesian Triplet Loss
    config = BayesianTripletConfig(
        margin=0.1,
        uncertainty_weight=0.1,
        adaptive_margin=True,
        triplet_mining='hardest',
        distance_metric='euclidean',
        uncertainty_type='gaussian',
        temperature=1.0
    )
    criterion = BayesianTripletLoss(config, device='cuda')
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    for epoch in range(10):
        # Example data (replace with your actual data)
        batch_size = 32
        features = torch.randn(batch_size, 784).cuda()
        labels = torch.randint(0, 10, (batch_size,)).cuda()
        
        # Forward pass
        embeddings, uncertainties = model(features)
        
        # Compute loss
        loss = criterion(embeddings, uncertainties, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Get loss components for analysis
        components = criterion.get_loss_components(embeddings, uncertainties, labels)
        
        print(f"Epoch {epoch}")
        print(f"  Total Loss: {loss.item():.4f}")
        print(f"  Uncertainty Reg: {components['uncertainty_regularization']:.4f}")
        print(f"  Mean Uncertainty: {components['mean_uncertainty']:.4f}")
        print()


def example_different_configurations():
    """Example showing different configuration options."""
    
    configurations = {
        'gaussian_euclidean': {
            'margin': 0.1,
            'uncertainty_weight': 0.1,
            'adaptive_margin': True,
            'triplet_mining': 'hardest',
            'distance_metric': 'euclidean',
            'uncertainty_type': 'gaussian'
        },
        'laplace_cosine': {
            'margin': 0.2,
            'uncertainty_weight': 0.2,
            'adaptive_margin': True,
            'triplet_mining': 'semihard',
            'distance_metric': 'cosine',
            'uncertainty_type': 'laplace'
        },
        'uniform_manhattan': {
            'margin': 0.15,
            'uncertainty_weight': 0.05,
            'adaptive_margin': False,
            'triplet_mining': 'all',
            'distance_metric': 'manhattan',
            'uncertainty_type': 'uniform'
        }
    }
    
    # Test each configuration
    batch_size = 16
    embedding_dim = 128
    embeddings = torch.randn(batch_size, embedding_dim).cuda()
    uncertainties = torch.rand(batch_size, embedding_dim).cuda() * 0.1
    labels = torch.randint(0, 5, (batch_size,)).cuda()
    
    for config_name, config_dict in configurations.items():
        print(f"\nTesting {config_name} configuration:")
        print(f"Config: {config_dict}")
        
        criterion = create_bayesian_triplet_loss(config_dict, device='cuda')
        loss = criterion(embeddings, uncertainties, labels)
        components = criterion.get_loss_components(embeddings, uncertainties, labels)
        
        print(f"Loss: {loss.item():.4f}")
        print(f"Components: {components}")


def example_uncertainty_analysis():
    """Example of analyzing uncertainty patterns."""
    
    config = BayesianTripletConfig(
        margin=0.1,
        uncertainty_weight=0.1,
        adaptive_margin=True,
        triplet_mining='hardest',
        distance_metric='euclidean',
        uncertainty_type='gaussian'
    )
    
    criterion = BayesianTripletLoss(config, device='cuda')
    
    # Test with different uncertainty levels
    batch_size = 16
    embedding_dim = 128
    embeddings = torch.randn(batch_size, embedding_dim).cuda()
    labels = torch.randint(0, 5, (batch_size,)).cuda()
    
    uncertainty_levels = [0.01, 0.1, 0.5, 1.0]
    
    print("Uncertainty Analysis:")
    print("=" * 50)
    
    for uncertainty_level in uncertainty_levels:
        uncertainties = torch.ones(batch_size, embedding_dim).cuda() * uncertainty_level
        
        loss = criterion(embeddings, uncertainties, labels)
        components = criterion.get_loss_components(embeddings, uncertainties, labels)
        
        print(f"Uncertainty Level: {uncertainty_level}")
        print(f"  Total Loss: {loss.item():.4f}")
        print(f"  Uncertainty Reg: {components['uncertainty_regularization']:.4f}")
        print(f"  Mean Uncertainty: {components['mean_uncertainty']:.4f}")
        print()


def example_comparison_with_traditional():
    """Example comparing Bayesian Triplet Loss with traditional triplet loss."""
    
    from bayesian_triplet_loss import TraditionalTripletLoss
    
    # Create both loss functions
    bayesian_config = BayesianTripletConfig(
        margin=0.1,
        uncertainty_weight=0.0,  # Disable uncertainty regularization for fair comparison
        adaptive_margin=False,   # Disable adaptive margin for fair comparison
        triplet_mining='hardest',
        distance_metric='euclidean',
        uncertainty_type='gaussian'
    )
    
    bayesian_loss = BayesianTripletLoss(bayesian_config, device='cuda')
    traditional_loss = TraditionalTripletLoss(margin=0.1, mining='hardest')
    
    # Test data
    batch_size = 16
    embedding_dim = 128
    embeddings = torch.randn(batch_size, embedding_dim).cuda()
    uncertainties = torch.rand(batch_size, embedding_dim).cuda() * 0.1
    labels = torch.randint(0, 5, (batch_size,)).cuda()
    
    # Compute losses
    bayesian_loss_value = bayesian_loss(embeddings, uncertainties, labels)
    traditional_loss_value = traditional_loss(embeddings, labels)
    
    print("Comparison with Traditional Triplet Loss:")
    print("=" * 50)
    print(f"Bayesian Triplet Loss: {bayesian_loss_value.item():.4f}")
    print(f"Traditional Triplet Loss: {traditional_loss_value.item():.4f}")
    print(f"Difference: {abs(bayesian_loss_value.item() - traditional_loss_value.item()):.4f}")


if __name__ == "__main__":
    print("=== Bayesian Triplet Loss Examples ===\n")
    
    print("1. Basic usage:")
    try:
        bayesian_loss, bayesian_loss_2 = example_basic_usage()
        print("✓ Bayesian Triplet Loss instances created successfully\n")
    except Exception as e:
        print(f"✗ Basic usage failed: {e}\n")
    
    print("2. Training loop example:")
    try:
        example_training_loop()
        print("✓ Training loop completed successfully\n")
    except Exception as e:
        print(f"✗ Training loop failed: {e}\n")
    
    print("3. Different configurations:")
    try:
        example_different_configurations()
        print("✓ Configuration testing completed successfully\n")
    except Exception as e:
        print(f"✗ Configuration testing failed: {e}\n")
    
    print("4. Uncertainty analysis:")
    try:
        example_uncertainty_analysis()
        print("✓ Uncertainty analysis completed successfully\n")
    except Exception as e:
        print(f"✗ Uncertainty analysis failed: {e}\n")
    
    print("5. Comparison with traditional triplet loss:")
    try:
        example_comparison_with_traditional()
        print("✓ Comparison completed successfully\n")
    except Exception as e:
        print(f"✗ Comparison failed: {e}\n")
    
    print("=== Key Features of Bayesian Triplet Loss ===")
    print("1. Uncertainty-aware distance computation")
    print("2. Adaptive margin based on uncertainty")
    print("3. Multiple uncertainty models (Gaussian, Laplace, Uniform)")
    print("4. Multiple distance metrics (Euclidean, Cosine, Manhattan)")
    print("5. Flexible triplet mining strategies")
    print("6. Uncertainty regularization for better training")
    print("7. Comprehensive loss component analysis")
    print("8. Backward compatibility with traditional triplet loss") 
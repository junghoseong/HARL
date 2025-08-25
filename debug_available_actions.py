#!/usr/bin/env python3
"""Debug script to test available_actions handling."""

import torch
import numpy as np

def test_categorical_sampling():
    """Test Categorical sampling with available_actions mask."""
    
    # Test data
    batch_size = 2
    num_actions = 9
    
    # Create logits
    logits = torch.randn(batch_size, num_actions)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits:\n{logits}")
    
    # Create available_actions mask
    available_actions = torch.tensor([
        [1, 1, 0, 1, 0, 1, 0, 1, 1],  # Only actions 0,1,3,5,7,8 are available
        [0, 1, 1, 0, 1, 0, 1, 0, 1]   # Only actions 1,2,4,6,8 are available
    ], dtype=torch.float32)
    
    print(f"\nAvailable actions mask:\n{available_actions}")
    
    # Apply mask to logits (same as in Categorical.forward)
    masked_logits = logits.clone()
    masked_logits[available_actions == 0] = -1e10
    print(f"\nMasked logits:\n{masked_logits}")
    
    # Create categorical distribution
    from torch.distributions import Categorical
    dist = Categorical(logits=masked_logits)
    
    # Sample actions
    actions = dist.sample()
    print(f"\nSampled actions: {actions}")
    
    # Check if actions are valid
    for i in range(batch_size):
        action = actions[i].item()
        is_valid = available_actions[i, action].item() > 0
        print(f"Sample {i}: Action {action} is valid: {is_valid}")
        
        if not is_valid:
            print(f"  ERROR: Action {action} is not available!")
            print(f"  Available actions for sample {i}: {torch.where(available_actions[i] > 0)[0].tolist()}")

if __name__ == "__main__":
    test_categorical_sampling() 
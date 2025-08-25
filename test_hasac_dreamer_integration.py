#!/usr/bin/env python3
"""
Test script for HASAC + Dreamer hybrid integration.
"""

import sys
import os
import torch
import numpy as np

# Add HARL to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'harl'))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from harl.runners.hasac_dreamer_runner import HASACDreamerRunner
        print("✅ HASACDreamerRunner imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import HASACDreamerRunner: {e}")
        return False
    
    try:
        from harl.runners import RUNNER_REGISTRY
        print("✅ RUNNER_REGISTRY imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import RUNNER_REGISTRY: {e}")
        return False
    
    try:
        from harl.models.dreamer.DreamerLearner import DreamerLearner
        print("✅ DreamerLearner imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import DreamerLearner: {e}")
        return False
    
    return True

def test_registry():
    """Test that the hybrid algorithm is registered."""
    print("\nTesting registry...")
    
    try:
        from harl.runners import RUNNER_REGISTRY
        if "hasac_dreamer" in RUNNER_REGISTRY:
            print("✅ hasac_dreamer registered in RUNNER_REGISTRY")
            print(f"   Runner class: {RUNNER_REGISTRY['hasac_dreamer']}")
        else:
            print("❌ hasac_dreamer not found in RUNNER_REGISTRY")
            return False
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from harl.utils.configs_tools import get_defaults_yaml_args
        
        # Test loading hybrid config
        algo_args, env_args = get_defaults_yaml_args("hasac_dreamer", "smac")
        
        # Check for Dreamer-specific parameters
        dreamer_params = [
            "dreamer_train_interval",
            "imagination_ratio", 
            "dreamer_model_lr",
            "dreamer_actor_lr",
            "dreamer_value_lr"
        ]
        
        for param in dreamer_params:
            if param in algo_args["algo"]:
                print(f"✅ {param} found in config: {algo_args['algo'][param]}")
            else:
                print(f"❌ {param} not found in config")
                return False
        
        print("✅ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_runner_creation():
    """Test runner creation with mock data."""
    print("\nTesting runner creation...")
    
    try:
        from harl.runners.hasac_dreamer_runner import HASACDreamerRunner
        
        # Mock arguments
        args = {
            "algo": "hasac_dreamer",  # Use hybrid algorithm name
            "env": "smac", 
            "exp_name": "test"
        }
        
        algo_args = {
            "model": {
                "hidden_sizes": [256, 256],
                "activation_func": "relu",
                "use_feature_normalization": True,
                "final_activation_func": "tanh",
                "initialization_method": "orthogonal_",
                "gain": 0.01,
                "critic_lr": 5e-4
            },
            "algo": {
                "lr": 5e-4,
                "polyak": 0.995,
                "gamma": 0.99,
                "dreamer_train_interval": 10,
                "imagination_ratio": 0.5,
                "dreamer_model_lr": 2e-4,
                "dreamer_actor_lr": 5e-4,
                "dreamer_value_lr": 5e-4,
                "dreamer_buffer_size": 100000,
                "dreamer_batch_size": 32,
                "dreamer_model_batch_size": 32,
                "dreamer_seq_length": 50,
                "dreamer_horizon": 15,
                "dreamer_entropy": 0.001,
                "dreamer_gamma": 0.99,
                "dreamer_expl_decay": 0.99998,
                "dreamer_expl_noise": 0.1,
                "dreamer_expl_min": 0.001,
                "dreamer_env_type": "starcraft",
                "auto_alpha": True,
                "alpha": 0.2,
                "target_entropy": -1,
                "use_policy_active_masks": True,
                "use_value_active_masks": True,
                "policy_freq": 2,
                "share_param": False,
                "fixed_order": False,
                "use_huber_loss": True,
                "huber_delta": 10.0,
                "alpha_lr": 3e-4,
                "n_step": 5
            },
            "train": {
                "n_rollout_threads": 2,
                "episode_length": 60,
                "num_env_steps": 1000000,
                "batch_size": 64,
                "buffer_size": 500000,
                "warmup_steps": 1000,
                "train_interval": 1,
                "log_interval": 10,
                "save_interval": 100,
                "eval_interval": 100,
                "model_dir": None,
                "use_proper_time_limits": True
            },
            "eval": {
                "use_eval": True,
                "n_eval_rollout_threads": 1,
                "eval_episodes": 10
            },
            "render": {
                "use_render": False,
                "render_episodes": 1
            },
            "device": {
                "cuda": False,
                "torch_threads": 4
            },
            "seed": {
                "seed": 1,
                "seed_specify": True
            },
            "logger": {
                "log_dir": "wandb"
            }
        }
        
        env_args = {
            "map_name": "3m",
            "n_agents": 3,
            "episode_limit": 60,
            "state_type": "FP"  # Add missing state_type parameter
        }
        
        # Mock environment
        class MockEnv:
            def __init__(self):
                self.observation_space = [[64, [2, 14], [3, 5], [1, 4], [1, 17]]] * 3
                self.action_space = [[9]] * 3
                self.share_observation_space = [64] * 3
            
            def reset(self):
                return [np.random.randn(64) for _ in range(3)], [np.random.randn(64) for _ in range(3)], [np.ones(9) for _ in range(3)]
            
            def step(self, actions):
                return [np.random.randn(64) for _ in range(3)], [np.random.randn(64) for _ in range(3)], [np.random.randn() for _ in range(3)], [False] * 3, [{}] * 3, [np.ones(9) for _ in range(3)]
        
        # Mock environment creation functions
        import harl.utils.envs_tools
        harl.utils.envs_tools.make_train_env = lambda *args, **kwargs: MockEnv()
        harl.utils.envs_tools.make_eval_env = lambda *args, **kwargs: MockEnv()
        harl.utils.envs_tools.get_num_agents = lambda *args, **kwargs: 3
        
        # Mock config tools
        import harl.utils.configs_tools
        harl.utils.configs_tools.get_task_name = lambda *args, **kwargs: "test_task"
        harl.utils.configs_tools.init_dir = lambda *args, **kwargs: ("/tmp/test", "/tmp/test/log", "/tmp/test/save", None)
        harl.utils.configs_tools.save_config = lambda *args, **kwargs: None
        
        # Create runner
        runner = HASACDreamerRunner(args, algo_args, env_args)
        print("✅ HASACDreamerRunner created successfully")
        
        # Test components
        print(f"   - Dreamer learner: {type(runner.dreamer_learner)}")
        print(f"   - Dreamer controller: {type(runner.dreamer_controller)}")
        print(f"   - Dreamer config: {type(runner.dreamer_config)}")
        print(f"   - Dreamer train interval: {runner.dreamer_train_interval}")
        print(f"   - Imagination ratio: {runner.imagination_ratio}")
        
        return True
        
    except Exception as e:
        print(f"❌ Runner creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing HASAC + Dreamer Hybrid Integration")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_registry,
        test_config_loading,
        test_runner_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! HASAC + Dreamer hybrid integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the integration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
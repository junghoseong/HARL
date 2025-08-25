#!/usr/bin/env python3
"""
Example script to train HASAC + Dreamer hybrid in HARL.
This script demonstrates how to use the integrated Dreamer world model
to enhance HASAC's sample efficiency.
"""

import argparse
import os
import sys
import yaml
import torch

# Add the HARL directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from harl.runners import RUNNER_REGISTRY
from harl.utils.configs_tools import get_defaults_yaml_args, update_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train HASAC + Dreamer Hybrid")
    parser.add_argument(
        "--algo",
        type=str,
        default="hasac_dreamer",
        help="Algorithm name (default: hasac_dreamer)"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="smac",
        help="Environment name (default: smac)"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="hasac_dreamer_test",
        help="Experiment name (default: hasac_dreamer_test)"
    )
    parser.add_argument(
        "--algo_cfg",
        type=str,
        default="harl/configs/algos_cfgs/hasac_dreamer.yaml",
        help="Algorithm config file path"
    )
    parser.add_argument(
        "--env_cfg",
        type=str,
        default="harl/configs/envs_cfgs/smac.yaml",
        help="Environment config file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (default: cpu)"
    )
    
    # Environment-specific arguments
    parser.add_argument(
        "--map_name",
        type=str,
        help="Map name for SMAC environment (e.g., 3m, 8m, corridor)"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario name for MPE environment"
    )
    parser.add_argument(
        "--n_agents",
        type=int,
        help="Number of agents"
    )
    parser.add_argument(
        "--episode_limit",
        type=int,
        help="Episode length limit"
    )
    
    # Dreamer-specific arguments
    parser.add_argument(
        "--dreamer_train_interval",
        type=int,
        default=10,
        help="Train Dreamer every N steps (default: 10)"
    )
    parser.add_argument(
        "--imagination_ratio",
        type=float,
        default=0.5,
        help="Ratio of imagined data in training (default: 0.5)"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Initialize configuration
    algo_args, env_args = get_defaults_yaml_args(args.algo, args.env)
    
    # Update configuration with command line arguments
    algo_args["seed"]["seed"] = args.seed
    if args.device == "cuda" and torch.cuda.is_available():
        algo_args["device"]["cuda"] = True
    else:
        algo_args["device"]["cuda"] = False
    
    # Update environment-specific arguments
    if args.map_name:
        env_args["map_name"] = args.map_name
    if args.scenario:
        env_args["scenario"] = args.scenario
    if args.n_agents:
        env_args["n_agents"] = args.n_agents
    if args.episode_limit:
        env_args["episode_limit"] = args.episode_limit
    
    # Update Dreamer-specific arguments
    algo_args["algo"]["dreamer_train_interval"] = args.dreamer_train_interval
    algo_args["algo"]["imagination_ratio"] = args.imagination_ratio
    
    # Create command line arguments dict
    cmd_args = {
        "algo": args.algo,
        "env": args.env,
        "exp_name": args.exp_name
    }
    
    # Get the appropriate runner
    if args.algo not in RUNNER_REGISTRY:
        raise ValueError(f"Algorithm {args.algo} not found in RUNNER_REGISTRY")
    
    runner_class = RUNNER_REGISTRY[args.algo]
    runner = runner_class(cmd_args, algo_args, env_args)
    
    # Start training
    print(f"Starting training with {args.algo} on {args.env}")
    print(f"Experiment name: {args.exp_name}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Dreamer train interval: {args.dreamer_train_interval}")
    print(f"Imagination ratio: {args.imagination_ratio}")
    
    try:
        runner.run()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        # Save the final model
        runner.save()
        print("Training completed and model saved")


if __name__ == "__main__":
    main() 
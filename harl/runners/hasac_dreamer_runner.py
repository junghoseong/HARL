"""HASAC + Dreamer hybrid runner for enhanced sample efficiency."""
import os
import time
import torch
import numpy as np
import setproctitle
from harl.common.valuenorm import ValueNorm
from torch.distributions import Categorical
from harl.utils.trans_tools import _t2n
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from harl.utils.models_tools import init_device
from harl.utils.configs_tools import init_dir, save_config, get_task_name
from harl.runners.off_policy_ha_runner import OffPolicyHARunner
from harl.models.dreamer.DreamerLearner import DreamerLearner
from harl.models.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
from harl.models.dreamer.DreamerMemory import DreamerMemory
from harl.models.dreamer.DreamerController import DreamerController
from harl.models.dreamer.DreamerAgentConfig import DreamerConfig
from harl.models.dreamer.environments import Env


class HASACDreamerRunner(OffPolicyHARunner):
    """HASAC + Dreamer hybrid runner for enhanced sample efficiency.
    
    Architecture:
    - HASAC: Heterogeneous actors (separate per agent) + Centralized critic (shared)
    - Dreamer: Centralized world model, actor, and critic (shared across all agents)
    """

    def __init__(self, args, algo_args, env_args):
        """Initialize the HASACDreamerRunner class."""
        # Temporarily change algo to "hasac" for base initialization
        original_algo = args["algo"]
        args["algo"] = "hasac"
        
        # Initialize HASAC components first
        super().__init__(args, algo_args, env_args)
        
        # Restore original algo name
        args["algo"] = original_algo
        
        # Initialize Dreamer components
        self._init_dreamer_components()
        
        # Training parameters for Dreamer integration
        self.dreamer_train_interval = algo_args["algo"].get("dreamer_train_interval", 10)
        self.imagination_ratio = algo_args["algo"].get("imagination_ratio", 0.5)  # 50% imagined data
        self.dreamer_warmup_steps = algo_args["algo"].get("dreamer_warmup_steps", 1000)
        
        # Track Dreamer training
        self.dreamer_total_it = 0
        self.last_dreamer_train = 0
        
        # Architecture flags
        self.use_dreamer_imagination = algo_args["algo"].get("use_dreamer_imagination", True)
        self.dreamer_centralized = algo_args["algo"].get("dreamer_centralized", True)

    def _init_dreamer_components(self):
        """Initialize centralized Dreamer components."""
        # Initialize Dreamer configuration
        self.dreamer_config = DreamerLearnerConfig()
        self.dreamer_config.DEVICE = str(self.device)
        
        # Handle different observation space formats
        if isinstance(self.envs.observation_space[0], list):
            self.dreamer_config.IN_DIM = self.envs.observation_space[0][0]
        else:
            self.dreamer_config.IN_DIM = self.envs.observation_space[0].shape[0]
        
        # Handle different action space formats
        if isinstance(self.envs.action_space[0], list):
            self.dreamer_config.ACTION_SIZE = self.envs.action_space[0][0]
        else:
            self.dreamer_config.ACTION_SIZE = self.envs.action_space[0].n if hasattr(self.envs.action_space[0], 'n') else self.envs.action_space[0].shape[0]
        
        # Update config with algo_args
        self.dreamer_config.MODEL_LR = float(self.algo_args["algo"].get("dreamer_model_lr", 2e-4))
        self.dreamer_config.ACTOR_LR = float(self.algo_args["algo"].get("dreamer_actor_lr", 5e-4))
        self.dreamer_config.VALUE_LR = float(self.algo_args["algo"].get("dreamer_value_lr", 5e-4))
        self.dreamer_config.CAPACITY = int(self.algo_args["algo"].get("dreamer_buffer_size", 100000))
        self.dreamer_config.BATCH_SIZE = int(self.algo_args["algo"].get("dreamer_batch_size", 32))
        self.dreamer_config.MODEL_BATCH_SIZE = int(self.algo_args["algo"].get("dreamer_model_batch_size", 32))
        self.dreamer_config.SEQ_LENGTH = int(self.algo_args["algo"].get("dreamer_seq_length", 50))
        self.dreamer_config.HORIZON = int(self.algo_args["algo"].get("dreamer_horizon", 15))
        self.dreamer_config.ENTROPY = float(self.algo_args["algo"].get("dreamer_entropy", 0.001))
        self.dreamer_config.GAMMA = float(self.algo_args["algo"].get("dreamer_gamma", 0.99))
        self.dreamer_config.EXPL_DECAY = float(self.algo_args["algo"].get("dreamer_expl_decay", 0.99998))
        self.dreamer_config.EXPL_NOISE = float(self.algo_args["algo"].get("dreamer_expl_noise", 0.1))
        self.dreamer_config.EXPL_MIN = float(self.algo_args["algo"].get("dreamer_expl_min", 0.001))
        
        # Set environment type
        env_type_str = self.algo_args["algo"].get("dreamer_env_type", "starcraft")
        if env_type_str == "starcraft":
            self.dreamer_config.ENV_TYPE = Env.STARCRAFT
        elif env_type_str == "flatland":
            self.dreamer_config.ENV_TYPE = Env.FLATLAND
        else:
            self.dreamer_config.ENV_TYPE = Env.STARCRAFT
        
        # Initialize centralized Dreamer learner
        self.dreamer_learner = DreamerLearner(self.dreamer_config)
        
        # Initialize centralized Dreamer controller
        self.dreamer_controller = DreamerController(self.dreamer_config)

    def train(self):
        """Train both HASAC and Dreamer components with proper architecture."""
        # Train Dreamer world model periodically
        if self.total_it % self.dreamer_train_interval == 0:
            self._train_dreamer()
        
        # Get training data
        real_data = self.buffer.sample()
        
        # Generate imagined data if enabled
        imagined_data = None
        if self.use_dreamer_imagination and self.dreamer_total_it > 0:
            imagined_data = self._generate_imagined_data()
        
        # Combine real and imagined data
        combined_data = self._combine_data(real_data, imagined_data)
        
        # Train HASAC with combined data
        self._train_hasac(combined_data)

    def _train_dreamer(self):
        """Train centralized Dreamer world model."""
        if len(self.dreamer_learner.replay_buffer) < self.dreamer_config.MIN_BUFFER_SIZE:
            return
        
        # Train world model
        for i in range(self.dreamer_config.MODEL_EPOCHS):
            samples = self.dreamer_learner.replay_buffer.sample(self.dreamer_config.MODEL_BATCH_SIZE)
            self.dreamer_learner.train_model(samples)
        
        # Train agent through imagination
        for i in range(self.dreamer_config.EPOCHS):
            samples = self.dreamer_learner.replay_buffer.sample(self.dreamer_config.BATCH_SIZE)
            self.dreamer_learner.train_agent(samples)
        
        # Update controller with new parameters
        params = self.dreamer_learner.params()
        self.dreamer_controller.receive_params(params)
        
        self.dreamer_total_it += 1
        self.last_dreamer_train = self.total_it

    def _generate_imagined_data(self):
        """Generate imagined data using centralized Dreamer."""
        if self.dreamer_total_it == 0:
            # Dreamer not trained yet, return empty data
            return None
        
        imagined_transitions = []
        batch_size = self.algo_args["train"]["batch_size"]
        
        # Sample initial states from buffer
        if len(self.dreamer_learner.replay_buffer) > 0:
            samples = self.dreamer_learner.replay_buffer.sample(min(batch_size, len(self.dreamer_learner.replay_buffer)))
            
            # Use centralized Dreamer controller to generate imagined rollouts
            for i in range(batch_size):
                if i < len(samples):
                    obs = samples[i]['observation']
                    imagined_rollout = self._imagine_rollout(obs)
                    imagined_transitions.extend(imagined_rollout)
        
        if not imagined_transitions:
            return None
        
        # Convert to HASAC format
        return self._convert_imagined_to_hasac_format(imagined_transitions)

    def _imagine_rollout(self, initial_obs, horizon=10):
        """Generate imagined rollout using centralized Dreamer."""
        rollout = []
        obs = initial_obs.clone()
        
        for step in range(horizon):
            # Get action from centralized Dreamer controller
            action = self.dreamer_controller.step(obs.unsqueeze(0), None, None)
            action = action.squeeze(0)
            
            # Use world model to predict next state and reward
            with torch.no_grad():
                # This is a simplified version - in practice, you'd use the full world model
                # For now, we'll use a simple approximation
                next_obs = obs + 0.1 * torch.randn_like(obs)  # Simplified dynamics
                reward = torch.tensor([0.0], device=self.device)  # Simplified reward
                done = torch.tensor([False], device=self.device)
            
            rollout.append({
                'obs': obs,
                'action': action,
                'reward': reward,
                'done': done,
                'next_obs': next_obs
            })
            
            obs = next_obs
            
            if done.item():
                break
        
        return rollout

    def _convert_imagined_to_hasac_format(self, imagined_transitions):
        """Convert imagined transitions to HASAC format."""
        if not imagined_transitions:
            return None
        
        # Extract data
        obs_list = [t['obs'] for t in imagined_transitions]
        actions_list = [t['action'] for t in imagined_transitions]
        rewards_list = [t['reward'] for t in imagined_transitions]
        dones_list = [t['done'] for t in imagined_transitions]
        next_obs_list = [t['next_obs'] for t in imagined_transitions]
        
        # Stack into tensors
        obs = torch.stack(obs_list)
        actions = torch.stack(actions_list)
        rewards = torch.stack(rewards_list)
        dones = torch.stack(dones_list)
        next_obs = torch.stack(next_obs_list)
        
        # Convert to HASAC format (simplified)
        # In practice, you'd need to handle the full HASAC data format
        return {
            'sp_obs': [obs],  # List for each agent
            'sp_actions': [actions],
            'sp_reward': rewards,
            'sp_done': dones,
            'sp_next_obs': [next_obs],
            'sp_share_obs': obs,  # Simplified
            'sp_next_share_obs': next_obs,
            'sp_available_actions': None,
            'sp_next_available_actions': None,
            'sp_valid_transition': torch.ones_like(dones),
            'sp_term': torch.zeros_like(dones),
            'sp_gamma': torch.full_like(dones, self.algo_args["algo"]["gamma"])
        }

    def get_actions(self, obs, available_actions=None, add_random=True):
        """Override get_actions to ensure proper handling of available_actions for SMAC."""
        # Since we inherit from OffPolicyHARunner and use HASAC actors, 
        # we should use the same logic as the original HASAC
        actions = []
        for agent_id in range(self.num_agents):
            if (
                available_actions is not None and 
                len(np.array(available_actions).shape) == 3
            ):  # (n_threads, n_agents, action_number)
                # Ensure available_actions is properly formatted
                agent_available_actions = available_actions[:, agent_id]
                actions.append(
                    _t2n(
                        self.actor[agent_id].get_actions(
                            obs[:, agent_id],
                            agent_available_actions,
                            add_random,
                        )
                    )
                )
            else:  # (n_threads, ) of None
                actions.append(
                    _t2n(
                        self.actor[agent_id].get_actions(
                            obs[:, agent_id], stochastic=add_random
                        )
                    )
                )
        return np.array(actions).transpose(1, 0, 2)

    def _combine_data(self, real_data, imagined_data):
        """Combine real and imagined data for training."""
        if imagined_data is None:
            return real_data
        
        # Simple combination: alternate between real and imagined data
        # In practice, you might want more sophisticated combination strategies
        combined_data = {}
        
        for key in real_data.keys():
            if key in imagined_data and imagined_data[key] is not None:
                if isinstance(real_data[key], list):
                    # Handle per-agent data
                    combined_data[key] = []
                    for agent_id in range(len(real_data[key])):
                        real_agent_data = real_data[key][agent_id]
                        imagined_agent_data = imagined_data[key][agent_id] if isinstance(imagined_data[key], list) else imagined_data[key]
                        
                        # Combine with imagination ratio
                        combined = torch.cat([real_agent_data, imagined_agent_data], dim=0)
                        combined_data[key].append(combined)
                else:
                    # Handle shared data
                    real_data_tensor = real_data[key]
                    imagined_data_tensor = imagined_data[key]
                    combined = torch.cat([real_data_tensor, imagined_data_tensor], dim=0)
                    combined_data[key] = combined
            else:
                combined_data[key] = real_data[key]
        
        return combined_data

    def _train_hasac(self, data):
        """Train HASAC with the provided data."""
        # Extract data
        (
            sp_share_obs,
            sp_obs,
            sp_actions,
            sp_available_actions,
            sp_reward,
            sp_done,
            sp_valid_transition,
            sp_term,
            sp_next_share_obs,
            sp_next_obs,
            sp_next_available_actions,
            sp_gamma,
        ) = data
        
        # Train critic
        self.critic.turn_on_grad()
        next_actions = []
        next_logp_actions = []
        for agent_id in range(self.num_agents):
            next_action, next_logp_action = self.actor[agent_id].get_actions_with_logprobs(
                sp_next_obs[agent_id],
                sp_next_available_actions[agent_id] if sp_next_available_actions is not None else None,
            )
            next_actions.append(next_action)
            next_logp_actions.append(next_logp_action)
        
        self.critic.train(
            sp_share_obs,
            sp_actions,
            sp_reward,
            sp_done,
            sp_valid_transition,
            sp_term,
            sp_next_share_obs,
            next_actions,
            next_logp_actions,
            sp_gamma,
            self.value_normalizer,
        )
        self.critic.turn_off_grad()
        
        # Train actors
        if self.total_it % self.policy_freq == 0:
            actions = []
            logp_actions = []
            with torch.no_grad():
                for agent_id in range(self.num_agents):
                    action, logp_action = self.actor[agent_id].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id] if sp_available_actions is not None else None,
                    )
                    actions.append(action)
                    logp_actions.append(logp_action)
            
            # Train each agent
            if self.fixed_order:
                agent_order = list(range(self.num_agents))
            else:
                agent_order = list(np.random.permutation(self.num_agents))
            
            for agent_id in agent_order:
                self.actor[agent_id].turn_on_grad()
                
                # Recompute actions and log probs for this agent
                actions[agent_id], logp_actions[agent_id] = self.actor[agent_id].get_actions_with_logprobs(
                    sp_obs[agent_id],
                    sp_available_actions[agent_id] if sp_available_actions is not None else None,
                )
                
                # Compute actor loss
                if self.state_type == "EP":
                    logp_action = logp_actions[agent_id]
                    actions_t = torch.cat(actions, dim=-1)
                elif self.state_type == "FP":
                    logp_action = torch.tile(logp_actions[agent_id], (self.num_agents, 1))
                    actions_t = torch.tile(torch.cat(actions, dim=-1), (self.num_agents, 1))
                
                value_pred = self.critic.get_values(sp_share_obs, actions_t)
                
                if self.algo_args["algo"]["use_policy_active_masks"]:
                    if self.state_type == "EP":
                        actor_loss = -torch.sum(
                            (value_pred - self.alpha[agent_id] * logp_action) * sp_valid_transition[agent_id]
                        ) / sp_valid_transition[agent_id].sum()
                    elif self.state_type == "FP":
                        valid_transition = torch.tile(torch.tensor(sp_valid_transition[agent_id], dtype=torch.float32), (self.num_agents, 1))
                        actor_loss = -torch.sum(
                            (value_pred - self.alpha[agent_id] * logp_action) * valid_transition
                        ) / valid_transition.sum()
                else:
                    actor_loss = -torch.mean(value_pred - self.alpha[agent_id] * logp_action)
                
                # Update actor
                self.actor[agent_id].actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor[agent_id].actor_optimizer.step()
                self.actor[agent_id].turn_off_grad()
                
                # Update alpha if auto_alpha is enabled
                if self.algo_args["algo"]["auto_alpha"]:
                    log_prob = logp_actions[agent_id].detach() + self.target_entropy[agent_id]
                    alpha_loss = -(self.log_alpha[agent_id] * log_prob).mean()
                    self.alpha_optimizer[agent_id].zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer[agent_id].step()
                    self.alpha[agent_id] = torch.exp(self.log_alpha[agent_id].detach())
                
                # Update actions for next agent
                actions[agent_id], _ = self.actor[agent_id].get_actions_with_logprobs(
                    sp_obs[agent_id],
                    sp_available_actions[agent_id] if sp_available_actions is not None else None,
                )
            
            # Update critic alpha
            if self.algo_args["algo"]["auto_alpha"]:
                self.critic.update_alpha(logp_actions, np.sum(self.target_entropy))
            
            # Soft updates (only for critic, not actors)
            self.critic.soft_update()

    def store_transitions(self, obs, share_obs, actions, rewards, dones, infos, available_actions):
        """Store transitions in both HASAC buffer and Dreamer buffer."""
        # Store in HASAC buffer (original behavior)
        super().store_transitions(obs, share_obs, actions, rewards, dones, infos, available_actions)
        
        # Store in Dreamer buffer
        rollout_data = {
            'observation': torch.tensor(obs, dtype=torch.float32),
            'action': torch.tensor(actions, dtype=torch.float32),
            'reward': torch.tensor(rewards, dtype=torch.float32),
            'done': torch.tensor(dones, dtype=torch.float32),
            'fake': torch.zeros_like(torch.tensor(dones, dtype=torch.float32)),
            'last': torch.zeros_like(torch.tensor(dones, dtype=torch.float32)),
            'avail_action': torch.tensor(available_actions, dtype=torch.float32) if available_actions is not None else None
        }
        
        self.dreamer_learner.replay_buffer.append(
            rollout_data['observation'],
            rollout_data['action'],
            rollout_data['reward'],
            rollout_data['done'],
            rollout_data['fake'],
            rollout_data['last'],
            rollout_data['avail_action']
        )

    def save(self):
        """Save both HASAC and Dreamer models."""
        # Save HASAC models
        super().save()
        
        # Save Dreamer models
        dreamer_save_path = os.path.join(self.save_dir, "dreamer")
        os.makedirs(dreamer_save_path, exist_ok=True)
        
        # Save Dreamer learner
        learner_params = self.dreamer_learner.params()
        torch.save(learner_params, os.path.join(dreamer_save_path, "learner.pt"))
        
        # Save Dreamer controller
        controller_state = {
            'actor': self.dreamer_controller.actor.state_dict(),
            'model': self.dreamer_controller.model.state_dict()
        }
        torch.save(controller_state, os.path.join(dreamer_save_path, "controller.pt"))

    def restore(self):
        """Restore both HASAC and Dreamer models."""
        # Restore HASAC models
        super().restore()
        
        # Restore Dreamer models
        dreamer_model_dir = os.path.join(self.algo_args["train"]["model_dir"], "dreamer")
        
        # Restore Dreamer learner
        learner_path = os.path.join(dreamer_model_dir, "learner.pt")
        if os.path.exists(learner_path):
            learner_params = torch.load(learner_path, map_location=self.device)
            self.dreamer_learner.model.load_state_dict(learner_params['model'])
            self.dreamer_learner.actor.load_state_dict(learner_params['actor'])
            self.dreamer_learner.critic.load_state_dict(learner_params['critic'])
        
        # Restore Dreamer controller
        controller_path = os.path.join(dreamer_model_dir, "controller.pt")
        if os.path.exists(controller_path):
            controller_state = torch.load(controller_path, map_location=self.device)
            self.dreamer_controller.actor.load_state_dict(controller_state['actor'])
            self.dreamer_controller.model.load_state_dict(controller_state['model']) 
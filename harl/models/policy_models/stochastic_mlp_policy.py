import torch
import torch.nn as nn
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.act import ACTLayer


class StochasticMlpPolicy(nn.Module):
    """Stochastic policy model that only uses MLP network. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticMlpPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(StochasticMlpPolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]

        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)
        self.act = ACTLayer(
            action_space,
            self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,
        )

        self.to(device)

    def forward(self, obs, available_actions=None, stochastic=True):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            stochastic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
        """
        obs = check(obs).to(**self.tpdv)
        deterministic = not stochastic
        
        # Handle case where available_actions might be a boolean (stochastic parameter)
        if available_actions is not None and not isinstance(available_actions, bool):
            available_actions = check(available_actions).to(**self.tpdv)
        else:
            available_actions = None

        actor_features = self.base(obs)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )
        
        # Safety check: ensure actions are valid if available_actions is provided
        if available_actions is not None and not deterministic:
            # Convert available_actions to boolean mask if needed
            if isinstance(available_actions, torch.Tensor):
                mask = available_actions.bool()
            else:
                mask = torch.tensor(available_actions, dtype=torch.bool, device=self.tpdv["device"])
            
            # For each sample, ensure the action is valid
            for i in range(actions.shape[0]):
                if not mask[i, actions[i]]:
                    # Find a valid action
                    valid_actions = torch.where(mask[i])[0]
                    if len(valid_actions) > 0:
                        actions[i] = valid_actions[torch.randint(0, len(valid_actions), (1,))]
        
        return actions

    def get_logits(self, obs, available_actions=None):
        """Get action logits from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) input to network.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                      (if None, all actions available)
        Returns:
            action_logits: (torch.Tensor) logits of actions for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        
        # Handle case where available_actions might be a boolean
        if available_actions is not None and not isinstance(available_actions, bool):
            available_actions = check(available_actions).to(**self.tpdv)
        else:
            available_actions = None

        actor_features = self.base(obs)

        return self.act.get_logits(actor_features, available_actions)

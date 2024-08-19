from gymnasium import spaces
import torch as th
import numpy as np
from torch.nn import functional as F
from typing import NamedTuple

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples


class RolloutBufferSamplesStep(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    non_zero_action_counter: th.Tensor
    episodic_step: th.Tensor


class RegPPO(PPO):
    def __init__(self, *args, l2_coef=None, l1_coef=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.l2_coef = None if l2_coef == 0.0 else l2_coef
        self.l1_coef = None if l1_coef == 0.0 else l1_coef

        assert self.l2_coef is None or self.l1_coef is None

    def train(self):
        """
        Subclasses the train method from Stable Baselines 3 PPO.
        This function adds
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                """regularization term"""
                l_loss = self.policy.l2_regularization() if self.l2_coef is not None else self.policy.l1_regularization()
                l_loss = self.l2_coef * l_loss if self.l2_coef is not None else self.l1_coef * l_loss

                """Add l2 loss in loss function"""
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                loss = loss + l_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/l2_loss" if self.l2_coef is not None else "train/l1_loss", l_loss.item())
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


class LagrangianPPO(PPO):
    def __init__(self, *args, constraint_fn=None, initial_lambda=0.1, lr_lambda=0.01, constraint_threshold=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint_fn = constraint_fn
        self.lambda_ = initial_lambda
        self.lr_lambda = lr_lambda
        self.constraint_threshold = constraint_threshold
        if self.n_envs > 1:
            self.non_zero_action_counter = np.zeros(self.n_envs)
        else:
            self.non_zero_action_counter = 0

    def _setup_model(self) -> None:
        super()._setup_model()

        self.rollout_buffer_class = RolloutBufferSteps

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                # Get step number in rollout buffer and non_zero_counter
                rollout_episodic_steps = rollout_data.episodic_step
                rollout_non_zero_action_counter = rollout_data.non_zero_action_counter

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # **Lagrangian Constraint Penalty Calculation**
                if self.constraint_fn is not None:
                    constraint_value = self.constraint_fn(actions,
                                                          rollout_episodic_steps,
                                                          rollout_non_zero_action_counter,)
                    constraint_violation = th.mean(constraint_value) - self.constraint_threshold
                    lagrangian_penalty = self.lambda_ * constraint_violation
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + lagrangian_penalty

                    # Update the Lagrange multiplier (gradient descent on lambda)
                    self.lambda_ += self.lr_lambda * constraint_violation.item()

                    if constraint_violation.item() <= 0:
                        self.lambda_ -= self.lr_lambda * 0.1  # Decay the multiplier slightly if the constraint is not violated
                    self.lambda_ = max(self.lambda_, 0)  # Ensure lambda is non-negative
                else:
                    # If no constraint function, use the standard loss
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                           self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self.logger.record("train/lagrange_multiplier", self.lambda_)  # Log the current value of lambda


def fertilization_action_constraint(actions,
                                    episodic_step,
                                    non_zero_action_counter,
                                    max_non_zero_actions=4,
                                    start_step=5,
                                    end_step=30,
                                    time_weight=0.6,
                                    n_weight=0.5):
    """
        Constraint function for RL agent.

        Parameters:
        - observations: The observations received from the environment.
        - actions: The actions taken by the agent.
        - step_number: The current step number in the episode.
        - max_non_zero_actions: Maximum number of non-zero actions allowed.
        - allowed_steps: List of steps at which certain actions are allowed.

        Returns:
        - constraint_value: The value representing the degree of constraint violation.
                            A positive value indicates a violation.
        """

    # Constraint 1: Number of non-zero actions that pass the defined threshold
    sampled_non_zero_actions = (actions > 0)
    constraint_1_violation = th.relu(non_zero_action_counter + sampled_non_zero_actions - max_non_zero_actions)

    # Constraint 2: Actions allowed only at specific step numbers
    invalid_steps_mask = (episodic_step < start_step) | (episodic_step >= end_step)
    non_zero_mask = actions != 0
    constraint_2_violation = th.logical_and(invalid_steps_mask, non_zero_mask)

    # The constraint value is a sum of the violations
    constraint_value = constraint_1_violation * n_weight + constraint_2_violation * time_weight

    return constraint_value


class RolloutBufferSteps(RolloutBuffer):

    episodic_step: np.ndarray
    non_zero_action_counter: np.ndarray

    def __init__(self, buffer_size, observation_space, action_space, device='cpu', gae_lambda=1, gamma=0.99, n_envs=1, **kwargs):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs, **kwargs)
        self.step_counter = np.zeros(n_envs, dtype=int)  # Initialize step counter for each environment

    def reset(self):
        """
        Reset the rollout buffer and step counters.
        """
        self.step_counter = np.zeros(self.n_envs, dtype=int)  # Reset step counters for each episode
        self.episodic_step = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)  # Reset rollout steps
        self.non_zero_action_counter = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)  # Reset counter
        super(RolloutBufferSteps, self).reset()

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        # Increment non_zero_action_counter
        for i in range(self.n_envs):
            if self.actions[self.pos][i] > 0:
                self.non_zero_action_counter[self.pos][i] += 1
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.episodic_step[self.pos] = self.step_counter
        self.step_counter += 1
        # Increment step counters for the environments where a step is added
        for idx in range(self.n_envs):
            if self.episode_starts[self.pos][idx]:
                self.step_counter[idx] = 0
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size = None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "non_zero_action_counter",
                "episodic_step",  # Adds info for step in episode
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env=None,
    ) -> RolloutBufferSamplesStep:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.non_zero_action_counter[batch_inds].flatten(),
            self.episodic_step[batch_inds].flatten(),
        )
        return RolloutBufferSamplesStep(*tuple(map(self.to_torch, data)))

    def compute_returns_and_advantage(self, last_values, dones):
        """
        Compute returns and advantage with step counting reset logic.
        Reset the step counter for environments where the episode has ended.
        """
        super(RolloutBufferSteps, self).compute_returns_and_advantage(last_values, dones)
        # Reset step counters for environments where the episode ended
        for env_idx in range(self.n_envs):
            if dones[env_idx]:
                self.step_counter[self.pos, env_idx] = 0
                self.non_zero_action_counter[self.pos, env_idx] = 0

    def get_episodic_step(self, env_idx=0):
        """
        Get the current step number for a specific environment.
        """
        return self.step_counter[env_idx]

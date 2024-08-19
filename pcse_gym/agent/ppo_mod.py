from gymnasium import spaces
import torch as th
import numpy as np
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.buffers import RolloutBuffer


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
    def __init__(self, *args, initial_lambda=0.1, lr_lambda=0.01, constraint_threshold=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint_fn = self.fertilization_action_constraint
        self.lambda_ = initial_lambda
        self.lr_lambda = lr_lambda
        self.constraint_threshold = constraint_threshold
        if self.n_envs > 1:
            self.non_zero_action_counter = np.zeros(self.n_envs)
        else:
            self.non_zero_action_counter = 0

    def _setup_model(self) -> None:
        super()._setup_model()

        if self.rollout_buffer_class is None:
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

                # Get step number in rollout buffer
                current_step = self.rollout_buffer.get_episode_step()

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
                    constraint_value = self.constraint_fn(rollout_data.observations, actions,
                                                          current_step)
                    constraint_violation = th.mean(constraint_value) - self.constraint_threshold
                    lagrangian_penalty = self.lambda_ * constraint_violation
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + lagrangian_penalty

                    # Update the Lagrange multiplier (gradient descent on lambda)
                    self.lambda_ += self.lr_lambda * constraint_violation.item()
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

    def fertilization_action_constraint(self,
                                        actions,
                                        step_number,
                                        max_non_zero_actions=4,
                                        start_step=5,
                                        end_step=30,
                                        time_weight=3,
                                        n_weight=2):
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


        # Constraint 1: Number of non-zero actions should be <= max_non_zero_actions

        if step_number == 0:
            self.non_zero_action_counter = np.zeros(self.n_envs) if self.n_envs > 1 else 0

        # Account for multiprocessing
        # TODO Add multiprocessing
        if self.n_envs > 1:
            constraint_1_violation = 0
            num_non_zero_actions = np.count_nonzero(actions)
            for idx in range(self.n_envs):
                self.non_zero_action_counter[idx] = ...  # look at this
            constraint_1_violation = max(0, num_non_zero_actions - max_non_zero_actions)
        else:
            num_non_zero_actions = np.count_nonzero(actions)
            self.non_zero_action_counter += 1 if num_non_zero_actions > 0 else 0
            constraint_1_violation = max(0, num_non_zero_actions - max_non_zero_actions)

        # Constraint 2: Actions allowed only at specific step numbers
        constraint_2_violation = 0
        if step_number not in range(start_step, end_step):
            constraint_2_violation = 1  # Violation if step number is not in the allowed list

        # The constraint value is a sum of the violations
        constraint_value = constraint_1_violation * n_weight + constraint_2_violation * time_weight

        return constraint_value


class RolloutBufferSteps(RolloutBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device='cpu', gae_lambda=1, gamma=0.99, n_envs=1, **kwargs):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs, **kwargs)
        self.step_counter = np.zeros(n_envs, dtype=int)  # Initialize step counter for each environment

    def reset(self):
        """
        Reset the rollout buffer and step counters.
        """
        super(RolloutBufferSteps, self).reset()
        self.step_counter = np.zeros(self.n_envs, dtype=int)  # Reset step counters for each environment

    def add(self, *args, **kwargs):
        """
        Add a new experience to the buffer.
        Also increment the step counter for the respective environment.
        """
        super(RolloutBufferSteps, self).add(*args, **kwargs)
        # Increment step counters for the environments where a step is added
        self.step_counter += 1
        for idx in range(self.n_envs):
            if self.episode_starts[self.pos][idx]:
                self.step_counter = 0

    def compute_returns_and_advantage(self, last_values, dones):
        """
        Compute returns and advantage with step counting reset logic.
        Reset the step counter for environments where the episode has ended.
        """
        super(RolloutBufferSteps, self).compute_returns_and_advantage(last_values, dones)
        # Reset step counters for environments where the episode ended
        self.step_counter[dones] = 0

    def get_episode_step(self, env_idx=0):
        """
        Get the current step number for a specific environment.
        """
        return self.step_counter[env_idx]

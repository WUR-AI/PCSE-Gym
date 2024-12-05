from gymnasium import spaces
import torch as th
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
from typing import NamedTuple, Union, Tuple
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, get_device
from stable_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm


class RolloutBufferSamplesStep(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    cost_advantages: th.Tensor
    returns: th.Tensor
    costs: th.Tensor
    cost_values: th.Tensor
    cost_returns: th.Tensor
    non_zero_action_counter: th.Tensor
    episodic_step: th.Tensor
    last_non_zero_action_step: th.Tensor
    nitrogen_efficiency: th.Tensor
    nitrogen_surplus: th.Tensor
    end_of_episode_mask: th.Tensor
    dvs: th.Tensor


class RolloutBufferNitrogenStep(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    cost_advantages: th.Tensor
    returns: th.Tensor
    costs: th.Tensor
    cost_values: th.Tensor
    cost_returns: th.Tensor
    non_zero_action_counter: th.Tensor
    episodic_step: th.Tensor
    last_non_zero_action_step: th.Tensor
    nitrogen_surplus: th.Tensor
    nitrogen_use_efficiency: th.Tensor


class LagrangianPPO(PPO):
    def __init__(self, *args, constraint_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint_fn = constraint_fn
        self.non_zero_action_counter = 0
        self.mean_approx_kl = None
        self.cost_vf_coef = 0.7
        self.lagrange = Lagrange(cost_limit=0.0, lagrangian_multiplier_init=0.001, lagrangian_multiplier_lr=0.0005,
                                 lagrangian_upper_bound=3)

    def _setup_model(self) -> None:
        super()._setup_model()

        # Ensure correct classes used
        self.rollout_buffer_class = RolloutBufferSteps
        self.policy_class = CostActorCriticPolicy

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

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
        pg_losses, value_losses, lagrangian_penalty, cost_value_losses = [], [], [], []
        constraint_1, constraint_2, constraint_3, constraint_nue, constraint_n_surplus = [], [], [], [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Calculate the mean episode cost from valid episodes
            episode_costs = self.rollout_buffer.get_costs_per_episode()
            valid_costs = [cost for env_costs in episode_costs for cost in env_costs]  # Flatten the list
            if valid_costs:  # Ensure there are valid episodes
                mean_ep_cost = np.mean(valid_costs)
                self.lagrange.update_lagrange_multiplier(mean_ep_cost)

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                # Get step number in rollout buffer and non_zero_counter
                rollout_episodic_steps = rollout_data.episodic_step
                rollout_non_zero_action_counter = rollout_data.non_zero_action_counter
                rollout_consecutive_non_zero = rollout_data.last_non_zero_action_step
                rollout_dvs = rollout_data.dvs
                rollout_nue = rollout_data.nitrogen_efficiency
                rollout_n_surp = rollout_data.nitrogen_surplus
                rollout_end_mask = rollout_data.end_of_episode_mask

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, cost_values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations,
                                                                                      actions)
                values = values.flatten()
                cost_values = cost_values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                cost_advantages = rollout_data.cost_advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Combine reward and cost advantages using the Lagrange multiplier
                lagrangian_multiplier = self.lagrange.lagrangian_multiplier
                combined_advantages = advantages - lagrangian_multiplier * cost_advantages
                combined_advantages /= (lagrangian_multiplier + 1)

                # clipped surrogate loss
                policy_loss_1 = combined_advantages * ratio
                policy_loss_2 = combined_advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # # Include the cost-based penalty using the updated Lagrange multiplier
                # lagrangian_multiplier = self.lagrange.lagrangian_multiplier
                # cost_penalty = lagrangian_multiplier * cost_advantages.mean()
                # policy_loss = policy_loss + cost_penalty

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

                # Value loss for costs
                cost_value_loss = F.mse_loss(rollout_data.cost_returns, cost_values)
                cost_value_losses.append(cost_value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # **Lagrangian Constraint Penalty Calculation**
                if self.constraint_fn is not None:
                    con1, con2, con3, nue_con, n_surp_con = self.constraint_fn(actions,
                                                                               rollout_episodic_steps,
                                                                               rollout_non_zero_action_counter,
                                                                               rollout_consecutive_non_zero,
                                                                               rollout_dvs,
                                                                               rollout_nue,
                                                                               rollout_n_surp,
                                                                               rollout_end_mask,
                                                                               )
                    con1_penalty, con2_penalty, con3_penalty, nue_penalty, n_surp_penalty = (th.mean(con1),
                                                                                             th.mean(con2),
                                                                                             th.mean(con3),
                                                                                             th.sum(nue_con if isinstance(nue_con, th.Tensor) else th.tensor(nue_con)),
                                                                                             th.sum(n_surp_con if isinstance(n_surp_con, th.Tensor) else th.tensor(n_surp_con)))
                    # constraint_violation = th.mean(constraint_value)  # - self.constraint_threshold
                    # constraint_violation = th.dot(self.lambda_, th.tensor([con1_violation, con2_violation, con3_violation]))

                    # self.lagrange.update_lagrange_multiplier(sum(constraint_violation))
                    # lagrangian_penalty = con1_penalty + con2_penalty + con3_penalty + nue_penalty + n_surp_penalty

                    # Calculate loss
                    loss = (policy_loss +
                            self.ent_coef * entropy_loss +
                            self.vf_coef * value_loss +
                            self.cost_vf_coef * cost_value_loss)  # + lagrangian_penalty

                    constraint_1.append(con1_penalty)
                    constraint_2.append(con2_penalty)
                    constraint_3.append(con3_penalty)
                    constraint_nue.append(nue_penalty)
                    constraint_n_surplus.append(n_surp_penalty)

                    # Update the Lagrange multiplier (gradient ascent on lambda)
                    # self.lambda_ = max(self.lambda_min, self.lambda_ + self.lr_lambda * constraint_violation.item() ** self.p)
                    # for i in range(len(self.lambda_)):
                    #     self.lambda_[i] = max(self.lambda_min,
                    #                    self.lambda_[i] + self.lr_lambda * constraint_violation[i].item() ** self.p)
                    #
                    #     # if constraint_violation[i].item() <= 0:
                    #     #     self.lambda_[i] -= self.lr_lambda * 1  # Decay the multiplier slightly if the constraint is not violated
                    #     self.lambda_[i] = max(self.lambda_[i], 0)  # Ensure lambda is non-negative
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

                # if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                #     continue_training = False
                #     if self.verbose >= 1:
                #         print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                #     break

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

        cost_prediction = explained_variance(self.rollout_buffer.cost_values.flatten(),
                                             self.rollout_buffer.cost_returns.flatten())

        self.mean_approx_kl = np.mean(approx_kl_divs)

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/cost_value_loss", np.mean(cost_value_losses))
        self.logger.record("train/lagrangian_multiplier", lagrangian_multiplier)
        self.logger.record("train/constraint_n_non_zero", np.mean(constraint_1))
        self.logger.record("train/constraint_weeks", np.mean(constraint_2))
        self.logger.record("train/constraint_nue", np.mean(constraint_nue))
        self.logger.record("train/constraint_n_surplus", np.mean(constraint_n_surplus))
        self.logger.record("train/constraint_consecutive_actions", np.mean(constraint_3))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/cost_prediction", cost_prediction)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        # self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        # self.logger.record("train/lagrange_multiplier", self.lambda_)  # Log the current value of lambda

    def augmented_lagrangian(self, cons):
        return cons + (self.p / 2) + th.linalg.norm(cons) ** 2

    def collect_rollouts(
            self,
            env,
            callback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, cost_values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Store N indicators
            nue = np.zeros(self.n_envs, dtype=np.float32)
            n_surplus = np.zeros(self.n_envs, dtype=np.float32)

            # Store DVS
            dvs = np.zeros(self.n_envs, dtype=np.float32)

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value
                if done:
                    nue[idx] = next(iter(infos[idx].get("NUE").values()))
                    n_surplus[idx] = next(iter(infos[idx].get("Nsurplus").values()))
                    rollout_buffer.add_nitrogen_info(nue, n_surplus)

            # add DVS to buffer
            for idx in range(self.n_envs):
                dvs[idx] = list(infos[idx].get("DVS").values())[-1]

            rollout_buffer.add_dvs(dvs)

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                cost_values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values, cost_values = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, last_cost_values=cost_values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True


def fertilization_action_constraint(actions,
                                    episodic_step,
                                    non_zero_action_counter,
                                    last_non_zero_action_step,
                                    dvs,
                                    nitrogen_efficiency,
                                    nitrogen_surplus,
                                    end_of_episode_mask,
                                    max_non_zero_actions=4,
                                    min_steps_between_actions=2,
                                    start_step=0.01,
                                    end_step=1,
                                    time_weight=50.,
                                    n_weight=5.,
                                    con_weight=3.,
                                    nue_threshold=(0.5, 0.9),
                                    n_surplus_threshold=(0, 40),
                                    ):
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
    constraint_1_violation = relu_func(non_zero_action_counter + sampled_non_zero_actions - max_non_zero_actions)

    # Constraint 2: Actions allowed only at specific step numbers based on DVS
    invalid_steps_mask = (dvs <= start_step) | (dvs >= end_step)
    non_zero_mask = actions != 0
    constraint_2_violation = np.logical_and(invalid_steps_mask, non_zero_mask)

    # Constraint 3: Ensure non-zero actions are spaced by at least `min_steps_between_actions`
    steps_since_last_non_zero = episodic_step - last_non_zero_action_step
    constraint_3_violation = np.logical_and(sampled_non_zero_actions,
                                            steps_since_last_non_zero < min_steps_between_actions)

    # The constraint value is a sum of the violations
    # constraint_value = (constraint_1_violation * n_weight +
    #                     constraint_2_violation * time_weight +
    #                     constraint_3_violation * con_weight)
    # constraint_2_violation = sampled_non_zero_actions * 0
    constraint_3_violation = sampled_non_zero_actions * 0

    # NUE and N Surplus constraints
    nue_violation = np.zeros_like(actions)
    n_surplus_violation = np.zeros_like(actions)
    if end_of_episode_mask.any():
        nue_violation = (nitrogen_efficiency < nue_threshold[0]) | (nitrogen_efficiency > nue_threshold[1]) | (nitrogen_efficiency != 0.0)
        n_surplus_violation = (nitrogen_surplus < nue_threshold[0]) | (nitrogen_surplus > n_surplus_threshold[1]) | (nitrogen_surplus != 0.0)
        nue_violation = nue_violation * np.ones_like(actions)
        n_surplus_violation = n_surplus_violation * np.ones_like(actions)

    return (constraint_1_violation * n_weight,
            constraint_2_violation * time_weight,
            constraint_3_violation * con_weight,
            nue_violation * 1.0,
            n_surplus_violation * 1.0)


def relu_func(x):
    return x * (x > 0)


def bool_func(x):
    return x > 0


class RolloutBufferSteps(RolloutBuffer):
    episodic_step: np.ndarray
    non_zero_action_counter: np.ndarray
    costs: np.ndarray
    costs_values: np.ndarray

    def __init__(self, buffer_size, observation_space, action_space, device='cpu', gae_lambda=1, gamma=1, n_envs=1,
                 **kwargs):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs, **kwargs)
        self.step_counter = np.zeros(n_envs, dtype=int)  # Initialize step counter for each environment
        self.last_non_zero_action_step = np.full((self.buffer_size, self.n_envs), -5,
                                                 dtype=int)  # For the consecutive non-zero action constraint
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.calculate_costs = fertilization_action_constraint
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_costs = [[] for _ in range(self.n_envs)]  # List to accumulate episode costs
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.nitrogen_efficiency = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.nitrogen_surplus = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.end_of_episode_mask = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.dvs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def reset(self):
        """
        Reset the rollout buffer and step counters.
        """
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.step_counter = np.zeros(self.n_envs, dtype=int)  # Reset step counters for each episode
        self.episodic_step = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)  # Reset rollout steps
        self.non_zero_action_counter = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)  # Reset counter
        self.last_non_zero_action_step = np.full((self.buffer_size, self.n_envs), -5,
                                                 dtype=int)  # Reset last non-zero action step counters
        self.episode_costs = [[] for _ in range(self.n_envs)]  # List to accumulate episode costs
        self.nitrogen_efficiency = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.nitrogen_surplus = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.end_of_episode_mask = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.dvs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super(RolloutBufferSteps, self).reset()

    def add_nitrogen_info(self, nue, n_surplus):
        """
        Add Nitrogen Use Efficiency (NUE) and N Surplus information to the buffer.
        """
        self.nitrogen_efficiency[self.pos] = nue
        self.nitrogen_surplus[self.pos] = n_surplus
        self.end_of_episode_mask[self.pos] = np.ones(self.n_envs, dtype=bool)

    def add_dvs(self, dvs):
        self.dvs[self.pos] = dvs

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            cost_value: th.Tensor,
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
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.cost_values[self.pos] = cost_value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()

        if not self.episode_starts[self.pos][0]:
            self.step_counter += 1
        else:
            self.step_counter = 0
        self.episodic_step[self.pos] = self.step_counter

        if len(self.episode_costs) == 0:
            self.episode_costs = [[] for _ in range(self.n_envs)]

        # Increment non_zero_action_counter and calculate costs and add N indicators
        for i in range(self.n_envs):
            if self.actions[self.pos][i] > 0:
                self.non_zero_action_counter[self.pos][i] = self.non_zero_action_counter[self.pos - 1][i] + 1
                self.last_non_zero_action_step[self.pos][i] = self.episodic_step[self.pos][i]
            # elif self.actions[self.pos][i] == 0 and self.last_non_zero_action_step[self.pos][i] == 0:
            else:
                self.non_zero_action_counter[self.pos][i] = self.non_zero_action_counter[self.pos - 1][i]
                self.last_non_zero_action_step[self.pos][i] = self.last_non_zero_action_step[self.pos - 1][i]
                # self.last_non_zero_action_step[self.pos][i] = self.step_counter[i]

            # Calculate costs for loss
            self.costs[self.pos][i] = np.sum(self.calculate_costs(self.actions[self.pos][i],
                                                                  self.episodic_step[self.pos][i],
                                                                  self.non_zero_action_counter[self.pos][i],
                                                                  self.last_non_zero_action_step[self.pos][i],
                                                                  self.dvs[self.pos][i],
                                                                  self.nitrogen_efficiency[self.pos][i],
                                                                  self.nitrogen_surplus[self.pos][i],
                                                                  self.end_of_episode_mask[self.pos][i], )
                                             )

            # If this is the start of a new episode, initialize tracking for costs
            if self.episode_starts[self.pos][i]:
                self.episode_costs[i].append([])

            # Ensure there's a list to append the cost to
            if len(self.episode_costs[i]) == 0:
                self.episode_costs[i].append([])

            # Accumulate the cost for the current episode
            self.episode_costs[i][-1].append(self.costs[self.pos][i])

        # Reset counters in new episodes
        for idx in range(self.n_envs):
            if self.episode_starts[self.pos][idx]:
                # self.step_counter[idx] = 0
                self.non_zero_action_counter[self.pos][idx] = 0
                self.last_non_zero_action_step[self.pos][idx] = 0
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get_costs_per_episode(self):
        """
        Get a list of total costs per episode for each environment, discarding episodes shorter than the mean length.

        Returns:
            List[List[float]]: A list where each element is a list of total costs for each valid episode in a particular environment.
        """
        # Calculate the mean episode length across all environments and episodes
        all_episode_lengths = [
            len(episode) for env_episodes in self.episode_costs for episode in env_episodes
        ]
        mean_episode_length = np.mean(all_episode_lengths)

        # Sum the costs within each episode for each environment, discarding short episodes
        total_costs_per_env = []
        for env_episodes in self.episode_costs:
            env_costs = []
            for episode in env_episodes:
                if len(episode) >= mean_episode_length:
                    env_costs.append(np.sum(episode))
            total_costs_per_env.append(env_costs)

        return total_costs_per_env

    def compute_returns_and_advantage(self, last_values: th.Tensor, last_cost_values: th.Tensor,
                                      dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()
        last_cost_values = last_cost_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        last_gae_lam_cost = 0

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
                next_cost_values = last_cost_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
                next_cost_values = self.cost_values[step + 1]

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            delta_cost = self.costs[step] + self.gamma * next_cost_values * next_non_terminal - self.cost_values[step]

            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            last_gae_lam_cost = delta_cost + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam_cost

            self.advantages[step] = last_gae_lam
            self.cost_advantages[step] = last_gae_lam_cost
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA

        self.returns = self.advantages + self.values
        self.cost_returns = self.cost_advantages + self.cost_values

    def get(self, batch_size=None):
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
                "cost_advantages",
                "returns",
                "costs",
                "cost_values",
                "cost_returns",
                "non_zero_action_counter",
                "episodic_step",  # Adds info for step in episode
                "last_non_zero_action_step",  #Track last time there were no actions
                "nitrogen_efficiency",
                "nitrogen_surplus",
                "end_of_episode_mask",
                "dvs"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
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
            self.cost_advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.costs[batch_inds].flatten(),
            self.cost_values[batch_inds].flatten(),
            self.cost_returns[batch_inds].flatten(),
            self.non_zero_action_counter[batch_inds].flatten(),
            self.episodic_step[batch_inds].flatten(),
            self.last_non_zero_action_step[batch_inds].flatten(),
            self.nitrogen_efficiency[batch_inds].flatten(),
            self.nitrogen_surplus[batch_inds].flatten(),
            self.end_of_episode_mask[batch_inds].flatten(),
            self.dvs[batch_inds].flatten(),
        )
        return RolloutBufferSamplesStep(*tuple(map(self.to_torch, data)))

    def get_episodic_step(self, env_idx=0):
        """
        Get the current step number for a specific environment.
        """
        return self.step_counter[env_idx]


class CostActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Initialize the policy network (actor)
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        # Initialize the value network (critic)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # Initialize the cost critic network
        self.cost_value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # Init weights: use orthogonal initialization
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                self.cost_value_net: 1,  # New cost value network initialization
            }

            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = CostMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        """
        Forward pass in all the networks (actor, value critic, and cost critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value, cost value, and log probability of the action
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf, latent_cf = self.mlp_extractor(features)
        else:
            pi_features, vf_features, cf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            latent_cf = self.mlp_extractor.forward_cost_critic(cf_features)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        cost_values = self.cost_value_net(latent_cf)  # Cost estimation

        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))

        return actions, values, cost_values, log_prob

    def evaluate_actions(self, obs, actions: th.Tensor):
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, cost value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf, latent_cf = self.mlp_extractor(features)
        else:
            pi_features, vf_features, cf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            latent_cf = self.mlp_extractor.forward_cost_critic(cf_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        cost_values = self.cost_value_net(latent_cf)  # Cost estimation
        entropy = distribution.entropy()

        return values, cost_values, log_prob, entropy

    def predict_values(self, obs):
        """
        Get the estimated values and cost values according to the current policy given the observations.

        :param obs: Observation
        :return: a tuple containing the estimated value and cost value.
        """
        features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)

        # Check if the features are returned as a tuple (for separate actor and critic features)
        if self.share_features_extractor:
            latent_vf = self.mlp_extractor(features)[1]  # Use the critic's features
            latent_cf = self.mlp_extractor(features)[2]
        else:
            pi_features, vf_features, cf_features = features
            latent_vf = self.mlp_extractor.forward_critic(vf_features)  # Use vf_features for the critic
            latent_cf = self.mlp_extractor.forward_cost_critic(cf_features)

        # Standard value estimation
        values = self.value_net(latent_vf)

        # Cost value estimation
        cost_values = self.cost_value_net(latent_cf)

        return values, cost_values


# Lagrange class taken from https://github.com/PKU-Alignment/safety-gymnasium,
# paper Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark
class Lagrange:
    """Lagrange multiplier for constrained optimization.

    Args:
        cost_limit: the cost limit
        lagrangian_multiplier_init: the initial value of the lagrangian multiplier
        lagrangian_multiplier_lr: the learning rate of the lagrangian multiplier
        lagrangian_upper_bound: the upper bound of the lagrangian multiplier

    Attributes:
        cost_limit: the cost limit
        lagrangian_multiplier_lr: the learning rate of the lagrangian multiplier
        lagrangian_upper_bound: the upper bound of the lagrangian multiplier
        _lagrangian_multiplier: the lagrangian multiplier
        lambda_range_projection: the projection function of the lagrangian multiplier
        lambda_optimizer: the optimizer of the lagrangian multiplier
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
            self,
            cost_limit: float,
            lagrangian_multiplier_init: float,
            lagrangian_multiplier_lr: float,
            lagrangian_upper_bound: Union[float, None] = None,
    ) -> None:
        """Initialize an instance of :class:`Lagrange`."""
        self.cost_limit: float = cost_limit
        self.lagrangian_multiplier_lr: float = lagrangian_multiplier_lr
        self.lagrangian_upper_bound: Union[float, None] = lagrangian_upper_bound

        init_value = max(lagrangian_multiplier_init, 0.0)
        self._lagrangian_multiplier: th.nn.Parameter = th.nn.Parameter(
            th.as_tensor(init_value),
            requires_grad=True,
        )
        self.lambda_range_projection: th.nn.ReLU = th.nn.ReLU()
        # fetch optimizer from PyTorch optimizer package
        self.lambda_optimizer: th.optim.Optimizer = th.optim.Adam(
            [
                self._lagrangian_multiplier,
            ],
            lr=lagrangian_multiplier_lr,
        )

    @property
    def lagrangian_multiplier(self) -> th.Tensor:
        """The lagrangian multiplier.

        Returns:
            the lagrangian multiplier
        """
        return self.lambda_range_projection(self._lagrangian_multiplier).detach().item()

    def compute_lambda_loss(self, mean_ep_cost: float) -> th.Tensor:
        """Compute the loss of the lagrangian multiplier.

        Args:
            mean_ep_cost: the mean episode cost

        Returns:
            the loss of the lagrangian multiplier
        """
        return -self._lagrangian_multiplier * (mean_ep_cost - self.cost_limit)

    def update_lagrange_multiplier(self, Jc: float) -> None:
        """Update the lagrangian multiplier.

        Args:
            Jc: the mean episode cost

        Returns:
            the loss of the lagrangian multiplier
        """
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(Jc)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self._lagrangian_multiplier.data.clamp_(
            0.0,
            self.lagrangian_upper_bound,
        )  # enforce: lambda in [0, inf]


class CostMlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy, value network, and cost value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>], cf=[<list of layer sizes>])``:
       to specify the amount and size of the layers in the policy, value, and cost nets individually.
       If it is missing any of the keys (pi, vf, cf), zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
       in the policy, value, and cost nets are the same. Same as ``dict(vf=int_list, pi=int_list, cf=int_list)``
       where int_list is the same for the actor, critic, and cost networks.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy, value, and cost networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def __init__(
            self,
            feature_dim: int,
            net_arch,
            activation_fn,
            device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net = []
        value_net = []
        cost_value_net = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim
        last_layer_dim_cf = feature_dim

        # Save dimensions of layers in policy, value, and cost value nets
        if isinstance(net_arch, dict):
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
            cf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the cost value network
        else:
            pi_layers_dims = vf_layers_dims = cf_layers_dims = net_arch

        # Build the policy network
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim

        # Build the value network
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Build the cost value network
        for curr_layer_dim in cf_layers_dims:
            cost_value_net.append(nn.Linear(last_layer_dim_cf, curr_layer_dim))
            cost_value_net.append(activation_fn())
            last_layer_dim_cf = curr_layer_dim

        # Save dim, used to create the distributions and final layers
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.latent_dim_cf = last_layer_dim_cf

        # Create networks
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)
        self.cost_value_net = nn.Sequential(*cost_value_net).to(device)

    def forward(self, features: th.Tensor):
        """
        :return: latent_policy, latent_value, latent_cost_value of the specified network.
            If all layers are shared, then `latent_policy == latent_value == latent_cost_value`
        """
        return (
            self.forward_actor(features),
            self.forward_critic(features),
            self.forward_cost_critic(features)
        )

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

    def forward_cost_critic(self, features: th.Tensor) -> th.Tensor:
        return self.cost_value_net(features)


class RolloutBufferNitrogen(RolloutBufferSteps):
    episodic_step: np.ndarray
    non_zero_action_counter: np.ndarray
    costs: np.ndarray
    costs_values: np.ndarray

    def __init__(self, buffer_size, observation_space, action_space, device='cpu', gae_lambda=1, gamma=0.99, n_envs=1,
                 **kwargs):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs, **kwargs)
        self.nitrogen_efficiency = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.nitrogen_surplus = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def reset(self):
        """
        Reset the rollout buffer and step counters.
        """
        super(RolloutBufferNitrogen, self).reset()
        self.nitrogen_efficiency = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.nitrogen_surplus = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add_nitrogen_info(self, nue, n_surplus):
        """
        Add Nitrogen Use Efficiency (NUE) and N Surplus information to the buffer.
        """
        self.nitrogen_efficiency[self.pos] = nue
        self.nitrogen_surplus[self.pos] = n_surplus

    def add(self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            cost_value: th.Tensor,
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
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.cost_values[self.pos] = cost_value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()

        if not self.episode_starts[self.pos][0]:
            self.step_counter += 1
        else:
            self.step_counter = 0
        self.episodic_step[self.pos] = self.step_counter

        if len(self.episode_costs) == 0:
            self.episode_costs = [[] for _ in range(self.n_envs)]

        # Increment non_zero_action_counter and calculate costs
        for i in range(self.n_envs):
            if self.actions[self.pos][i] > 0:
                self.non_zero_action_counter[self.pos][i] = self.non_zero_action_counter[self.pos - 1][i] + 1
                self.last_non_zero_action_step[self.pos][i] = self.episodic_step[self.pos][i]
            # elif self.actions[self.pos][i] == 0 and self.last_non_zero_action_step[self.pos][i] == 0:
            else:
                self.non_zero_action_counter[self.pos][i] = self.non_zero_action_counter[self.pos - 1][i]
                self.last_non_zero_action_step[self.pos][i] = self.last_non_zero_action_step[self.pos - 1][i]
                # self.last_non_zero_action_step[self.pos][i] = self.step_counter[i]

            self.costs[self.pos][i] = np.sum(self.calculate_costs(self.actions[self.pos][i],
                                                                  self.episodic_step[self.pos][i],
                                                                  self.non_zero_action_counter[self.pos][i],
                                                                  self.last_non_zero_action_step[self.pos][i]))

            # If this is the start of a new episode, initialize tracking for costs
            if self.episode_starts[self.pos][i]:
                self.episode_costs[i].append([])

            # Ensure there's a list to append the cost to
            if len(self.episode_costs[i]) == 0:
                self.episode_costs[i].append([])

            # Accumulate the cost for the current episode
            self.episode_costs[i][-1].append(self.costs[self.pos][i])

        # Reset counters in new episodes
        for idx in range(self.n_envs):
            if self.episode_starts[self.pos][idx]:
                # self.step_counter[idx] = 0
                self.non_zero_action_counter[self.pos][idx] = 0
                self.last_non_zero_action_step[self.pos][idx] = 0
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get_costs_per_episode(self):
        """
        Get a list of total costs per episode for each environment, discarding episodes shorter than the mean length.

        Returns:
            List[List[float]]: A list where each element is a list of total costs for each valid episode in a particular environment.
        """
        # Calculate the mean episode length across all environments and episodes
        all_episode_lengths = [
            len(episode) for env_episodes in self.episode_costs for episode in env_episodes
        ]
        mean_episode_length = np.mean(all_episode_lengths)

        # Sum the costs within each episode for each environment, discarding short episodes
        total_costs_per_env = []
        for env_episodes in self.episode_costs:
            env_costs = []
            for episode in env_episodes:
                if len(episode) >= mean_episode_length:
                    env_costs.append(np.sum(episode))
            total_costs_per_env.append(env_costs)

        return total_costs_per_env

    def get(self, batch_size=None):
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
                "cost_advantages",
                "returns",
                "costs",
                "cost_values",
                "cost_returns",
                "non_zero_action_counter",
                "episodic_step",  # Adds info for step in episode
                "last_non_zero_action_step",  #Track last time there were no actions
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
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
            self.cost_advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.costs[batch_inds].flatten(),
            self.cost_values[batch_inds].flatten(),
            self.cost_returns[batch_inds].flatten(),
            self.non_zero_action_counter[batch_inds].flatten(),
            self.episodic_step[batch_inds].flatten(),
            self.last_non_zero_action_step[batch_inds].flatten(),
        )
        return RolloutBufferSamplesStep(*tuple(map(self.to_torch, data)))

    def get_episodic_step(self, env_idx=0):
        """
        Get the current step number for a specific environment.
        """
        return self.step_counter[env_idx]

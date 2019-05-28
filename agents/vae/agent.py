import torch
from torch import nn
from rainy.agents import PpoAgent
from rainy.envs import State
from rainy.lib.rollout import RolloutSampler
from rainy.prelude import Array
from .loss import GammaVaeLoss


class VaePpoAgent(PpoAgent):
    def _one_step(self, states: Array[State]) -> Array[State]:
        with torch.no_grad():
            policy, value = self.net.p_and_v(self.penv.extract(states))
        next_states, rewards, done, info = self.penv.step(policy.action().squeeze().cpu().numpy())
        self.episode_length += 1
        self.rewards += rewards
        self.report_reward(done, info)
        self.storage.push(next_states, rewards, done, policy=policy, value=value)
        return next_states

    def nstep(self, states: Array[State]) -> Array[State]:
        for _ in range(self.config.nsteps):
            states = self._one_step(states)

        with torch.no_grad():
            next_value = self.net.value(self.penv.extract(states))

        if self.config.use_gae:
            gamma, tau = self.config.discount_factor, self.config.gae_tau
            self.storage.calc_gae_returns(next_value, gamma, tau)
        else:
            self.storage.calc_ac_returns(next_value, self.config.discount_factor)
        p, v, e, r, lt = 0.0, 0.0, 0.0, 0.0, 0.0
        for _ in range(self.config.ppo_epochs):
            sampler = RolloutSampler(
                self.storage,
                self.penv,
                self.config.ppo_minibatch_size,
                adv_normalize_eps=self.config.adv_normalize_eps,
            )
            for batch in sampler:
                vae, policy, value = self.net(batch.states)
                policy.set_action(batch.actions)
                policy_loss = self._policy_loss(policy, batch.advantages, batch.old_log_probs)
                value_loss = self._value_loss(value, batch.values, batch.returns)
                entropy_loss = policy.entropy().mean()
                recons_loss, latent_loss = self.config.vae_loss(vae, batch.states)
                self.optimizer.zero_grad()
                (policy_loss
                 + self.config.value_loss_weight * value_loss
                 + self.config.vae_loss_weight * (recons_loss + latent_loss)
                 - self.config.entropy_weight * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
                self.optimizer.step()
                p, v, e = p + policy_loss.item(), v + value_loss.item(), e + entropy_loss.item()
                r, lt = r + recons_loss.item(), lt + latent_loss.item()

        self.storage.reset()
        self.lr_cooler.lr_decay(self.optimizer)
        self.clip_eps = self.clip_cooler()
        p, v, e, r, lt = map(lambda x: x / float(self.num_updates), (p, v, e, r, lt))
        self.report_loss(policy_loss=p, value_loss=v,
                         entropy_loss=e, recons_loss=r, latent_loss=lt)
        if isinstance(self.config.vae_loss, GammaVaeLoss):
            self.config.vae_loss.update()
        return states

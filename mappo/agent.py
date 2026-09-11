import torch
import torch.nn.functional as F
from torch.optim import Adam

from .networks import Actor, CentralCritic

class MAPPOAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        state_dim,
        num_agents,
        actor_lr=3e-4,
        critic_lr=1e-3,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=8,
        minibatch_size=256,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.num_agents = num_agents
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs = epochs
        self.minibatch_size = minibatch_size

        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = CentralCritic(state_dim, num_agents).to(self.device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

    @torch.no_grad()
    def act(self, observations, action_masks=None, deterministic=False):
        obs = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )

        mask = None
        if action_masks is not None:
            mask = torch.as_tensor(
                action_masks, dtype=torch.bool, device=self.device
            )

        dist = self.actor.distribution(obs, mask)

        if deterministic:
            actions = torch.argmax(dist.logits, dim=-1)
        else:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            entropy.cpu().numpy(),
        )

    @torch.no_grad()
    def value(self, global_state):
        state = torch.as_tensor(
            global_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        return self.critic(state).squeeze(0).cpu().numpy()

    def update(self, batch):
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        agent_ids = torch.as_tensor(batch["agent_ids"], dtype=torch.long, device=self.device)
        masks = torch.as_tensor(batch["action_masks"], dtype=torch.bool, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(obs)
        actor_losses = []
        critic_losses = []
        entropies = []

        for _ in range(self.epochs):
            permutation = torch.randperm(n, device=self.device)

            for start in range(0, n, self.minibatch_size):
                mb = permutation[start:start + self.minibatch_size]

                dist = self.actor.distribution(obs[mb], masks[mb])
                new_logp = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp[mb])
                surr1 = ratio * advantages[mb]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                ) * advantages[mb]

                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss -= self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                values_all = self.critic(states[mb])
                values = values_all.gather(
                    1,
                    agent_ids[mb].unsqueeze(1)
                ).squeeze(1)

                critic_loss = F.mse_loss(values, returns[mb]) * self.value_coef

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy.item()))

        return {
            "actor_loss": sum(actor_losses) / max(len(actor_losses), 1),
            "critic_loss": sum(critic_losses) / max(len(critic_losses), 1),
            "entropy": sum(entropies) / max(len(entropies), 1),
        }

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path, map_location=None):
        checkpoint = torch.load(
            path,
            map_location=map_location or self.device,
            weights_only=False,
        )
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

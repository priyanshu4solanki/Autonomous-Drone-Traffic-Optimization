import argparse
import os
import random
import numpy as np
import torch

from environment.drone_env import DroneTrafficEnv
from mappo.agent import MAPPOAgent
from mappo.rollout import compute_gae

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_config = {
        "num_drones": args.drones,
        "max_steps": args.max_steps,
    }
    env = DroneTrafficEnv(env_config)

    obs, _ = env.reset(seed=args.seed)

    agent = MAPPOAgent(
        obs_dim=env.observation_space.shape[-1],
        action_dim=env.action_space.nvec[0],
        state_dim=env.global_state_dim,
        num_agents=env.num_drones,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        entropy_coef=args.entropy_coef,
        epochs=args.epochs,
        minibatch_size=args.minibatch,
    )

    os.makedirs("checkpoints", exist_ok=True)

    best_delivery = -1.0
    best_reward = -float("inf")

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode)

        observations, states, actions, logps, rewards, dones, values, action_masks = [], [], [], [], [], [], [], []
        episode_reward = np.zeros(env.num_drones, dtype=np.float32)

        for _ in range(env.config["max_steps"]):
            state = env.get_global_state()
            masks = env.get_action_masks()

            action, logp, entropy = agent.act(obs, masks)
            value = agent.value(state)

            next_obs, reward, terminated, truncated, info = env.step(action)

            observations.append(obs.copy())
            states.append(state.copy())
            actions.append(action.copy())
            logps.append(logp.copy())
            rewards.append(reward.copy())
            dones.append(np.full(env.num_drones, terminated or truncated, dtype=np.float32))
            values.append(value.copy())
            action_masks.append(masks.copy())

            episode_reward += reward
            obs = next_obs

            if terminated or truncated:
                break

        observations = np.asarray(observations, dtype=np.float32)
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        logps = np.asarray(logps, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        action_masks = np.asarray(action_masks, dtype=bool)

        advantages, returns = compute_gae(
            rewards, values, dones,
            gamma=args.gamma,
            lam=args.gae_lambda,
        )

        T, N = rewards.shape
        batch = {
            "obs": observations.reshape(T * N, -1),
            "states": np.repeat(states, N, axis=0),
            "actions": actions.reshape(T * N),
            "logp": logps.reshape(T * N),
            "advantages": advantages.reshape(T * N),
            "returns": returns.reshape(T * N),
            "agent_ids": np.tile(np.arange(N), T),
            "action_masks": action_masks.reshape(T * N, -1),
        }

        metrics = agent.update(batch)

        total_reward = float(np.sum(episode_reward))
        deliveries = info["deliveries"]
        collisions = info["collisions"]
        no_fly = info["no_fly_violations"]
        delivery_rate = deliveries / env.num_drones

        if delivery_rate > best_delivery or (delivery_rate == best_delivery and total_reward > best_reward):
            best_delivery = delivery_rate
            best_reward = total_reward
            agent.save("checkpoints/mappo_best.pt")

        if episode == 1 or episode % args.log_every == 0:
            print(
                f"Ep {episode:4d} | "
                f"Reward {total_reward:9.2f} | "
                f"Delivery {deliveries}/{N} | "
                f"Collisions {collisions:3d} | "
                f"NoFly {no_fly:3d} | "
                f"Entropy {metrics['entropy']:.3f}"
            )

    agent.save("checkpoints/mappo.pt")
    print("\nSaved final model to checkpoints/mappo.pt")
    print("Saved best model  to checkpoints/mappo_best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--drones", type=int, default=5, help="Number of drones")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--log-every", type=int, default=10)
    train(parser.parse_args())
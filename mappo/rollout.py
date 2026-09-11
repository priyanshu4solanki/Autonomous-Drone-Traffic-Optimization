import numpy as np

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)

    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_values = np.zeros(N, dtype=np.float32)
        else:
            next_values = values[t + 1]

        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        advantages[t] = last

    return advantages, advantages + values

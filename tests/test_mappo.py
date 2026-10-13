import numpy as np
import torch

from mappo.agent import MAPPOAgent
from mappo.route_guidance import choose_guided_actions
from environment.drone_env import DroneTrafficEnv


def make_agent(use_cnn=True):
    return MAPPOAgent(
        obs_dim=18,
        action_dim=6,
        state_dim=17,
        num_agents=2,
        epochs=1,
        minibatch_size=2,
        use_cnn=use_cnn,
    )


def test_cnn_actor_selects_actions_and_updates_from_local_maps():
    agent = make_agent()
    observations = np.zeros((2, 18), dtype=np.float32)
    local_maps = np.zeros((2, 6, 9, 9), dtype=np.float32)
    local_maps[:, 4, 4, 4] = 1
    action_masks = np.ones((2, 6), dtype=bool)

    _, _, _ = agent.act(
        observations,
        action_masks=action_masks,
        local_maps=local_maps,
    )

    np.testing.assert_allclose(agent.last_action_probs.sum(axis=1), 1.0, atol=1e-6)

    actions = np.array([0, 1], dtype=np.int64)
    with torch.no_grad():
        log_probs = agent.actor.distribution(
            torch.as_tensor(observations),
            torch.as_tensor(local_maps),
            torch.as_tensor(action_masks),
        ).log_prob(torch.as_tensor(actions)).numpy()

    old_weights = agent.actor.map_encoder[0].weight.detach().clone()
    metrics = agent.update({
        "obs": np.zeros((2, 18), dtype=np.float32),
        "local_maps": local_maps,
        "states": np.zeros((2, 17), dtype=np.float32),
        "actions": actions,
        "logp": log_probs,
        "advantages": np.array([1.0, -1.0], dtype=np.float32),
        "returns": np.zeros(2, dtype=np.float32),
        "agent_ids": np.array([0, 1]),
        "action_masks": action_masks,
    })

    assert np.isfinite(metrics["actor_loss"])
    assert not np.array_equal(old_weights.numpy(), agent.actor.map_encoder[0].weight.detach().numpy())


def test_cnn_checkpoint_round_trip(tmp_path):
    agent = make_agent()
    checkpoint = tmp_path / "cnn.pt"
    agent.save(checkpoint)

    restored = make_agent()
    restored.load(checkpoint)

    assert restored.use_cnn
    assert restored.actor.state_dict().keys() == agent.actor.state_dict().keys()


def test_legacy_mlp_checkpoint_remains_loadable(tmp_path):
    legacy = make_agent(use_cnn=False)
    checkpoint = tmp_path / "legacy.pt"
    legacy.save(checkpoint)

    restored = make_agent()
    restored.load(checkpoint)
    actions, _, _ = restored.act(
        np.zeros((2, 18), dtype=np.float32),
        action_masks=np.ones((2, 6), dtype=bool),
    )

    assert not restored.use_cnn
    assert actions.shape == (2,)


def test_route_guidance_uses_legal_actions_and_delivers():
    env = DroneTrafficEnv({
        "width": 15,
        "height": 15,
        "num_buildings": 4,
        "num_no_fly_zones": 2,
        "num_warehouses": 8,
        "num_delivery_points": 8,
        "num_drones": 3,
        "battery": 200,
        "max_steps": 200,
    })
    env.reset(seed=53)
    rng = np.random.default_rng(4)

    for _ in range(env.config["max_steps"]):
        masks = env.get_action_masks()
        probabilities = rng.random((env.num_drones, 6))
        actions = choose_guided_actions(env, probabilities, masks)
        _, _, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break

    assert info["deliveries"] == env.num_drones

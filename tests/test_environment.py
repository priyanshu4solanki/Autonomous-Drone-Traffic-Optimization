import numpy as np

from environment import DroneTrafficEnv

def test_reset_shapes_and_valid_points():
    env = DroneTrafficEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (env.num_drones, 18)
    assert np.isfinite(obs).all()
    for d in env.drones:
        assert not env.city.is_blocked(d.position.x, d.position.y)
        assert not env.city.is_blocked(d.destination.x, d.destination.y)

def test_action_masks_have_valid_actions():
    env = DroneTrafficEnv()
    env.reset(seed=1)
    masks = env.get_action_masks()
    assert masks.shape == (env.num_drones, 6)
    assert masks.any(axis=1).all()

def test_invalid_action_does_not_move_or_consume_battery():
    env = DroneTrafficEnv({"num_drones": 1})
    env.reset(seed=5)

    d = env.drones[0]
    d.position = type(d.position)(0, 0)
    old_position = d.position
    old_battery = d.battery

    # Force an outside-city move.
    env.step([1])
    assert d.position == old_position
    assert d.battery == old_battery

def test_delivery_reward():
    env = DroneTrafficEnv({"num_drones": 1})
    env.reset(seed=10)
    d = env.drones[0]
    d.position = d.destination
    _, reward, _, _, info = env.step([4])
    assert info["deliveries"] == 1
    assert reward[0] > 0

def test_gym_api():
    env = DroneTrafficEnv()
    obs, _ = env.reset(seed=0)
    result = env.step(env.action_space.sample())
    assert len(result) == 5

import numpy as np

from environment import DroneTrafficEnv
from environment.task_generator import TaskGenerator

def test_reset_shapes_and_valid_points():
    env = DroneTrafficEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (env.num_drones, 18)
    assert np.isfinite(obs).all()
    assert env.observation_space.contains(obs)
    for d in env.drones:
        assert not env.city.is_blocked(d.position.x, d.position.y)
        assert not env.city.is_blocked(d.destination.x, d.destination.y)

def test_action_masks_have_valid_actions():
    env = DroneTrafficEnv()
    env.reset(seed=1)
    masks = env.get_action_masks()
    assert masks.shape == (env.num_drones, 6)
    assert masks.any(axis=1).all()

def test_local_maps_encode_each_drone_view():
    env = DroneTrafficEnv({"num_drones": 2, "observation_radius": 2})
    env.reset(seed=1)

    maps = env.get_local_maps()
    assert maps.shape == (2, 6, 5, 5)
    assert np.isfinite(maps).all()
    assert (maps[:, 4, 2, 2] == 1).all()

def test_local_map_contains_destination_when_in_view():
    env = DroneTrafficEnv({"num_drones": 1, "observation_radius": 4})
    env.reset(seed=3)
    drone = env.drones[0]
    offset = 1 if drone.position.x + 1 < env.city.width else -1
    drone.destination = type(drone.position)(drone.position.x + offset, drone.position.y)

    local_map = env.get_local_maps()[0]
    assert local_map[5, 4, 4 + offset] == 1

def test_reset_assigns_distinct_start_and_destination_cells():
    for seed in range(20):
        env = DroneTrafficEnv({"num_drones": 10})
        env.reset(seed=seed)
        starts = [drone.position for drone in env.drones]
        destinations = [drone.destination for drone in env.drones]
        assert len(set(starts)) == env.num_drones
        assert len(set(destinations)) == env.num_drones
        assert set(starts).isdisjoint(destinations)

def test_reset_accepts_valid_user_selected_missions():
    env = DroneTrafficEnv({"num_drones": 2})
    env.reset(seed=17)
    tasks = [
        (env.city.warehouses[0], env.city.delivery_points[0]),
        (env.city.warehouses[1], env.city.delivery_points[1]),
    ]

    env.reset(seed=17, options={"tasks": tasks})

    assert [drone.position for drone in env.drones] == [task[0] for task in tasks]
    assert [drone.destination for drone in env.drones] == [task[1] for task in tasks]

def test_reset_rejects_duplicate_user_selected_missions():
    env = DroneTrafficEnv({"num_drones": 2})
    env.reset(seed=17)
    task = (env.city.warehouses[0], env.city.delivery_points[0])

    try:
        env.reset(seed=17, options={"tasks": [task, task]})
    except ValueError as error:
        assert "distinct" in str(error)
    else:
        raise AssertionError("Expected duplicate user missions to be rejected")

def test_joint_move_conflict_is_rejected_and_counted_once():
    env = DroneTrafficEnv({"num_drones": 2})
    env.reset(seed=5)
    left, right = env.drones
    x, y = next(
        (x, y)
        for x in range(1, env.city.width - 1)
        for y in range(env.city.height)
        if all(not env.city.is_blocked(cell_x, y) for cell_x in (x - 1, x, x + 1))
    )
    left.position = type(left.position)(x - 1, y)
    right.position = type(right.position)(x + 1, y)
    contested = type(left.position)(x, y)

    _, rewards, _, _, info = env.step([3, 2])

    assert sum(drone.position == contested for drone in env.drones) == 1
    assert info["collisions"] == 1
    assert info["collision_involvements"] == 1
    assert min(info["agent_rewards"]) < 0

def test_edge_swap_conflict_is_rejected():
    env = DroneTrafficEnv({"num_drones": 2})
    env.reset(seed=8)
    left, right = env.drones
    x, y = next(
        (x, y)
        for x in range(env.city.width - 1)
        for y in range(env.city.height)
        if not env.city.is_blocked(x, y) and not env.city.is_blocked(x + 1, y)
    )
    left.position = type(left.position)(x, y)
    right.position = type(right.position)(x + 1, y)
    left_before, right_before = left.position, right.position

    masks = env.get_action_masks()
    _, _, _, _, info = env.step([3, 2])

    assert left.position == left_before
    assert right.position == right_before
    assert not masks[0, 3] and not masks[1, 2]
    assert info["collisions"] == 0

def test_inactive_drone_position_remains_reserved():
    env = DroneTrafficEnv({"num_drones": 2})
    env.reset(seed=9)
    delivered, moving = env.drones
    x, y = next(
        (x, y)
        for x in range(env.city.width - 1)
        for y in range(env.city.height)
        if not env.city.is_blocked(x, y) and not env.city.is_blocked(x + 1, y)
    )
    delivered.position = type(delivered.position)(x + 1, y)
    delivered.destination = delivered.position
    moving.position = type(moving.position)(x, y)
    moving.destination = type(moving.position)(10, 10)

    env.step([4, 4])
    assert not delivered.active

    masks = env.get_action_masks()
    _, _, _, _, info = env.step([4, 3])
    assert moving.position == type(moving.position)(x, y)
    assert not masks[1, 3]
    assert info["collisions"] == 0

def test_random_actions_never_leave_drones_overlapping():
    for seed in range(5):
        env = DroneTrafficEnv({"num_drones": 8, "max_steps": 40})
        env.reset(seed=seed)
        for _ in range(env.config["max_steps"]):
            actions = env.action_space.sample()
            env.step(actions)
            positions = [drone.position for drone in env.drones]
            assert len(set(positions)) == env.num_drones
            if all(not drone.active for drone in env.drones):
                break

def test_task_generator_fails_clearly_if_fleet_cannot_fit():
    env = DroneTrafficEnv({"width": 4, "height": 4, "num_drones": 9})
    try:
        env.reset(seed=12)
    except ValueError as error:
        assert "distinct drone tasks" in str(error)
    else:
        raise AssertionError("Expected a clear error for an oversized fleet")

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
    assert reward > 0
    assert info["agent_rewards"][0] > 0

def test_gym_api():
    env = DroneTrafficEnv()
    obs, _ = env.reset(seed=0)
    _, reward, _, _, info = env.step(env.action_space.sample())
    assert isinstance(reward, float)
    assert info["agent_rewards"].shape == (env.num_drones,)

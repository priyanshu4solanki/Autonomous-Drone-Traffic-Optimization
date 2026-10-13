import argparse
import os
import numpy as np
import torch

from environment.drone_env import DroneTrafficEnv
from environment.obstacles import Point
from environment.renderer import Renderer
from mappo.agent import MAPPOAgent
from mappo.route_guidance import choose_guided_actions


def prompt_integer(label, default, minimum, maximum):
    while True:
        response = input(f"{label} [{default}]: ").strip()
        if not response:
            return default
        try:
            value = int(response)
        except ValueError:
            print("Enter a whole number.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"Choose a value from {minimum} to {maximum}.")


def show_locations(title, locations):
    print(f"\n{title}")
    for index, point in enumerate(locations, start=1):
        print(f"  {index:2}: ({point.x:2}, {point.y:2})", end="")
        if index % 5 == 0:
            print()
    if len(locations) % 5:
        print()


def prompt_missions(env):
    warehouses = env.city.warehouses
    delivery_points = env.city.delivery_points
    show_locations("Choose a warehouse number for each drone:", warehouses)
    show_locations("Choose a delivery-point number for each drone:", delivery_points)
    print("\nChoose distinct warehouse and delivery-point numbers for each drone.")

    missions = []
    used_sources = set()
    used_destinations = set()
    for drone_id in range(env.num_drones):
        while True:
            response = input(
                f"Drone {drone_id + 1} mission (warehouse, delivery point), "
                "for example 2,7: "
            ).strip()
            try:
                source_id, destination_id = (
                    int(value.strip()) for value in response.split(",")
                )
                source = warehouses[source_id - 1]
                destination = delivery_points[destination_id - 1]
            except (ValueError, IndexError):
                print("Enter two valid menu numbers separated by a comma.")
                continue

            if source in used_sources:
                print("That warehouse is already assigned; choose another.")
                continue
            if destination in used_destinations:
                print("That delivery point is already assigned; choose another.")
                continue

            missions.append((Point(source.x, source.y), Point(destination.x, destination.y)))
            used_sources.add(source)
            used_destinations.add(destination)
            break
    return missions


def evaluate(args):
    # ============================================================
    # CREATE ENVIRONMENT
    # ============================================================

    env = DroneTrafficEnv(
        config={
            "num_drones": args.drones,
            "max_steps": args.max_steps,
        }
    )

    # ============================================================
    # INITIAL RESET
    # ============================================================
    # This is important because ObservationBuilder is created
    # inside env.reset().

    observations, info = env.reset(seed=args.seed)
    custom_tasks = None
    if args.interactive:
        print("\nMAP LEGEND: B building | X no-fly zone | W warehouse")
        print("            D delivery point | C charging station")
        response = input("\nChoose custom warehouse/delivery points? [Y/n]: ").strip().lower()
        if response not in ("n", "no"):
            custom_tasks = prompt_missions(env)
            observations, info = env.reset(
                seed=args.seed,
                options={"tasks": custom_tasks},
            )

    # ============================================================
    # CREATE MAPPO AGENT
    # ============================================================

    global_state = env.get_global_state()

    agent = MAPPOAgent(
        obs_dim=env.observation_space.shape[-1],
        action_dim=6,
        state_dim=global_state.shape[0],
        num_agents=args.drones,
        device=args.device,
    )

    # ============================================================
    # LOAD TRAINED MODEL
    # ============================================================

    agent.load(args.model, map_location=args.device, load_critic=False)

    print()
    print("=" * 60)
    print("MAPPO EVALUATION")
    print("=" * 60)
    print(f"Model:          {args.model}")
    print(f"Episodes:       {args.episodes}")
    print(f"Drones:         {args.drones}")
    print(f"Max steps:      {args.max_steps}")
    print(f"Device:         {args.device}")
    print(f"Deterministic:  {args.deterministic}")
    print(
        f"Action selection: {'CNN with A* route guidance' if args.route_guidance else 'CNN/MLP policy'}"
    )
    print("=" * 60)
    print()

    # ============================================================
    # RENDERER
    # ============================================================

    renderer = None

    if args.render:
        renderer = Renderer(
            env,
            cell_size=args.cell_size,
            sidebar_width=args.sidebar_width,
            fps=args.fps,
            policy_type=(
                ("CNN" if agent.use_cnn else "MLP")
                + (" + route guidance" if args.route_guidance else "")
            ),
        )

    # ============================================================
    # EVALUATION METRICS
    # ============================================================

    episode_rewards = []
    episode_deliveries = []
    episode_collisions = []
    episode_collision_involvements = []
    episode_no_fly = []
    episode_battery = []
    episode_steps = []

    # ============================================================
    # EPISODE LOOP
    # ============================================================

    for episode in range(1, args.episodes + 1):

        # Reset environment
        observations, info = env.reset(
            seed=args.seed + episode - 1,
            options={"tasks": custom_tasks} if custom_tasks is not None else None,
        )

        # Reset renderer for new episode
        if renderer is not None:
            renderer.reset()

        done = False
        episode_reward = np.zeros(args.drones, dtype=np.float32)
        steps = 0

        # ========================================================
        # STEP LOOP
        # ========================================================

        while not done:

            # ----------------------------------------------------
            # Get decentralized local observations
            # ----------------------------------------------------

            observations = env.get_local_observations()

            # ----------------------------------------------------
            # Get legal action masks
            # ----------------------------------------------------

            action_masks = env.get_action_masks()
            local_maps = env.get_local_maps()

            # ----------------------------------------------------
            # Select actions using decentralized actors
            #
            # IMPORTANT:
            # MAPPOAgent uses:
            #
            #     local observation + map crop -> shared actor -> action
            #
            # The global state is NOT passed to act().
            #
            # The global state is only used by the centralized
            # critic during training.
            # ----------------------------------------------------

            actions, log_probs, entropy = agent.act(
                observations,
                action_masks=action_masks,
                deterministic=args.deterministic,
                local_maps=local_maps,
            )
            if args.route_guidance:
                actions = choose_guided_actions(
                    env,
                    agent.last_action_probs,
                    action_masks,
                )

            # ----------------------------------------------------
            # Render the current CNN input and selected actions before moving.
            if renderer is not None:
                can_advance = renderer.render(
                    actions=actions,
                    action_probabilities=agent.last_action_probs,
                )
                if not can_advance:
                    if not renderer.running:
                        done = True
                        break
                    continue

            # Execute actions
            # ----------------------------------------------------

            next_observations, _, terminated, truncated, info = env.step(
                actions
            )

            # ----------------------------------------------------
            # Accumulate rewards
            # ----------------------------------------------------

            episode_reward += info["agent_rewards"]

            steps += 1

            # ----------------------------------------------------
            # Check episode termination
            # ----------------------------------------------------

            done = terminated or truncated

        # ========================================================
        # COLLECT EPISODE METRICS
        # ========================================================

        deliveries = info.get(
            "deliveries",
            env.total_deliveries,
        )

        collisions = info.get(
            "collisions",
            sum(d.collisions for d in env.drones),
        )

        no_fly = info.get(
            "no_fly_violations",
            sum(d.no_fly_violations for d in env.drones),
        )

        battery = info.get(
            "battery_remaining",
            float(
                np.mean(
                    [d.battery for d in env.drones]
                )
            ),
        )

        total_reward = float(np.sum(episode_reward))

        episode_rewards.append(total_reward)
        episode_deliveries.append(deliveries)
        episode_collisions.append(collisions)
        episode_collision_involvements.append(info.get("collision_involvements", 0))
        episode_no_fly.append(no_fly)
        episode_battery.append(battery)
        episode_steps.append(steps)

        # ========================================================
        # PRINT EPISODE RESULT
        # ========================================================

        print(
            f"Episode {episode:4d} | "
            f"Reward: {total_reward:9.2f} | "
            f"Deliveries: {deliveries}/{args.drones} | "
            f"Conflicts: {collisions:3d} | "
            f"No-fly: {no_fly:3d} | "
            f"Battery: {battery:6.2f} | "
            f"Steps: {steps:4d}"
        )

        if renderer is not None and renderer.running:
            renderer.render()

        if args.interactive and renderer is not None:
            renderer.show_results(
                deliveries=deliveries,
                total=args.drones,
                conflicts=collisions,
                steps=steps,
            )

    # ============================================================
    # FINAL STATISTICS
    # ============================================================

    total_possible_deliveries = args.episodes * args.drones
    total_deliveries = sum(episode_deliveries)

    delivery_rate = (
        100.0 * total_deliveries / total_possible_deliveries
        if total_possible_deliveries > 0
        else 0.0
    )

    avg_reward = float(
        np.mean(episode_rewards)
    )

    avg_collisions = float(
        np.mean(episode_collisions)
    )
    avg_collision_involvements = float(
        np.mean(episode_collision_involvements)
    )

    avg_no_fly = float(
        np.mean(episode_no_fly)
    )

    avg_battery = float(
        np.mean(episode_battery)
    )

    avg_steps = float(
        np.mean(episode_steps)
    )

    # ============================================================
    # PRINT FINAL RESULTS
    # ============================================================

    print()
    print("=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"Delivery rate:       {delivery_rate:.3f}%"
    )

    print(
        f"Deliveries:          "
        f"{total_deliveries}/{total_possible_deliveries}"
    )

    print(
        f"Prevented conflicts/episode: {avg_collisions:.3f}"
    )

    print(f"Drone conflict involvements: {avg_collision_involvements:.3f}")

    print(
        f"No-fly/episode:      {avg_no_fly:.3f}"
    )

    print(
        f"Average reward:      {avg_reward:.3f}"
    )

    print(
        f"Average battery:     {avg_battery:.3f}"
    )

    print(
        f"Average steps:       {avg_steps:.2f}"
    )

    print("=" * 60)
    print()

    # ============================================================
    # CLOSE RENDERER
    # ============================================================

    if renderer is not None:
        renderer.close()

    env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MAPPO drone traffic model."
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/cnn/mappo_best.pt",
        help="Path to trained MAPPO checkpoint.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for fleet size and missions, then launch a visual demo.",
    )

    # ------------------------------------------------------------
    # Evaluation settings
    # ------------------------------------------------------------

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes.",
    )

    parser.add_argument(
        "--drones",
        type=int,
        default=5,
        help="Number of drones.",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Maximum steps per episode.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    # ------------------------------------------------------------
    # MAPPO inference
    # ------------------------------------------------------------

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic action selection.",
    )
    parser.add_argument(
        "--route-guidance",
        action="store_true",
        help="Constrain CNN choices to shortest safe routes using the simulator map.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device: cpu or cuda.",
    )

    # ------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------

    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable Pygame visualization.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=3,
        help="Rendering FPS.",
    )

    parser.add_argument(
        "--cell_size",
        type=int,
        default=28,
        help="Pixel size of each city grid cell.",
    )

    parser.add_argument(
        "--sidebar_width",
        type=int,
        default=440,
        help="Width of the renderer sidebar.",
    )

    args = parser.parse_args(argv)

    if args.interactive:
        print("=" * 62)
        print("AUTONOMOUS DRONE DELIVERY — INTERACTIVE SIMULATION")
        print("=" * 62)
        print("Simulation only; this does not control real drone hardware.")
        args.drones = prompt_integer("How many delivery drones?", 5, 1, 10)
        args.episodes = 1
        args.max_steps = prompt_integer("Maximum simulation steps?", 300, 20, 2000)
        args.seed = prompt_integer(
            "City seed (same seed reproduces the city)?", 42, 0, 2**31 - 1
        )
        args.fps = prompt_integer("Playback speed in frames per second?", 3, 1, 20)
        response = input(
            "Enable A* route guidance for the recommended demo? [Y/n]: "
        ).strip().lower()
        args.route_guidance = response not in ("n", "no")
        args.render = True
        args.deterministic = True
        print(f"\nLoading model: {args.model}")
        print("Controls: SPACE pause/resume, N single-step, +/- speed, ESC exit.")

    if not os.path.isfile(args.model):
        raise FileNotFoundError(
            f"Model checkpoint not found: {args.model}. "
            "Train one with `python train_mappo.py` or pass --model."
        )

    # ------------------------------------------------------------
    # Validate arguments
    # ------------------------------------------------------------

    if args.episodes <= 0:
        raise ValueError(
            "--episodes must be greater than 0"
        )

    if args.drones <= 0:
        raise ValueError(
            "--drones must be greater than 0"
        )

    if args.max_steps <= 0:
        raise ValueError(
            "--max_steps must be greater than 0"
        )

    if args.fps <= 0:
        raise ValueError(
            "--fps must be greater than 0"
        )

    evaluate(args)


if __name__ == "__main__":
    main()
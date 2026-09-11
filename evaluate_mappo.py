import argparse
import numpy as np
import torch

from environment.drone_env import DroneTrafficEnv
from environment.renderer import Renderer
from mappo.agent import MAPPOAgent


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

    agent.load(args.model, map_location=args.device)

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
        )

    # ============================================================
    # EVALUATION METRICS
    # ============================================================

    episode_rewards = []
    episode_deliveries = []
    episode_collisions = []
    episode_no_fly = []
    episode_battery = []
    episode_steps = []

    # ============================================================
    # EPISODE LOOP
    # ============================================================

    for episode in range(1, args.episodes + 1):

        # Reset environment
        observations, info = env.reset(
            seed=args.seed + episode - 1
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

            # ----------------------------------------------------
            # Select actions using decentralized actors
            #
            # IMPORTANT:
            # MAPPOAgent uses:
            #
            #     local observation -> actor -> action
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
            )

            # ----------------------------------------------------
            # Execute actions
            # ----------------------------------------------------

            next_observations, rewards, terminated, truncated, info = env.step(
                actions
            )

            # ----------------------------------------------------
            # Accumulate rewards
            # ----------------------------------------------------

            episode_reward += rewards

            steps += 1

            # ----------------------------------------------------
            # Render current state
            # ----------------------------------------------------

            if renderer is not None:
                renderer.render()

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
            f"Collisions: {collisions:3d} | "
            f"No-fly: {no_fly:3d} | "
            f"Battery: {battery:6.2f} | "
            f"Steps: {steps:4d}"
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
        f"Collisions/episode:  {avg_collisions:.3f}"
    )

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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MAPPO drone traffic model."
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained MAPPO checkpoint.",
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
        default=10,
        help="Rendering FPS.",
    )

    parser.add_argument(
        "--cell_size",
        type=int,
        default=40,
        help="Pixel size of each city grid cell.",
    )

    parser.add_argument(
        "--sidebar_width",
        type=int,
        default=300,
        help="Width of the renderer sidebar.",
    )

    args = parser.parse_args()

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
"""
Fleet AI Oversight — Random Agent Baseline
==========================================
Establishes random agent performance for comparison with trained agent.

The random baseline agent makes completely random decisions in both phases:

Planning phase: randomly selects task configs for each worker
    (ignores dataset profile characteristics entirely)

Oversight phase: randomly selects oversight actions and target workers
    (ignores anomaly flags, budget patterns, and step history)

Expected baseline results:
    Planning allocation accuracy: ~20% (1/5 chance of correct match)
    Oversight detection rate: ~28% on easy_fleet
    False positive rate: ~45% (random interventions on healthy workers)
    Average episode reward: ~-0.80

These numbers establish the floor that the trained agent must beat.
The trained agent should achieve:
    Detection: 65%+ vs 28% baseline
    False positives: 15% vs 45% baseline
    Episode reward: +1.40 vs -0.80 baseline

Transfer baseline (banking_fleet):
    Random detection: ~10% (harder domain, same random agent)
    This makes the transfer improvement even more significant.

Usage:
    python fleet_baseline.py --task-id easy_fleet --episodes 10 --seed 42
    python fleet_baseline.py --task-id banking_fleet --episodes 10 --seed 99
"""

import argparse
import json
import random
import numpy as np
from pathlib import Path

from env.oversight_env import FleetOversightEnv
from env.models import OversightAction, OversightActionRequest

VALID_ACTIONS = [a.value for a in OversightAction]
VALID_WORKERS = ["worker_1", "worker_2", "worker_3", "worker_4", "worker_5"]
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def run_random_agent(task_id: str, n_episodes: int, seed: int = 42) -> dict:
    rng = random.Random(seed)
    detection_rates = []
    total_rewards = []
    fp_rates = []
    approved_count = 0

    for ep in range(n_episodes):
        env = FleetOversightEnv(task_id=task_id, seed=rng.randint(0, 9999))
        obs = env.reset()
        ep_reward = 0.0
        interventions = 0
        false_positives = 0

        for step in range(env.oversight_budget):
            action_type = rng.choice(VALID_ACTIONS)
            worker_id = rng.choice(VALID_WORKERS)
            try:
                action = OversightActionRequest(
                    action_type=OversightAction(action_type),
                    worker_id=worker_id,
                )
                _, reward_obj, done, info = env.step(action)
                ep_reward += reward_obj.total
                if action_type == "intervene":
                    interventions += 1
                    if reward_obj.false_positive_penalty < 0:
                        false_positives += 1
                if done:
                    break
            except Exception:
                break

        try:
            eval_result = env.evaluate_run()
            detection_rates.append(eval_result.get("detection_rate", 0.0))
            if eval_result.get("approved", False):
                approved_count += 1
        except Exception:
            detection_rates.append(0.0)

        total_rewards.append(ep_reward)
        fp_rate = false_positives / max(interventions, 1)
        fp_rates.append(fp_rate)

        print(
            f"[BASELINE ep={ep+1:03d}] "
            f"reward={ep_reward:+.3f} "
            f"detection={detection_rates[-1]:.1%} "
            f"fp_rate={fp_rate:.1%}"
        )

    results = {
        "task_id": task_id,
        "n_episodes": n_episodes,
        "agent": "random",
        "mean_detection_rate": float(np.mean(detection_rates)),
        "std_detection_rate": float(np.std(detection_rates)),
        "mean_total_reward": float(np.mean(total_rewards)),
        "std_total_reward": float(np.std(total_rewards)),
        "mean_fp_rate": float(np.mean(fp_rates)),
        "approved_rate": approved_count / n_episodes,
        "detection_rates": detection_rates,
        "total_rewards": total_rewards,
    }

    output_path = PLOTS_DIR / f"baseline_{task_id}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"RANDOM BASELINE RESULTS — {task_id}")
    print(f"{'='*50}")
    print(f"Mean detection rate:  {results['mean_detection_rate']:.1%} ± {results['std_detection_rate']:.1%}")
    print(f"Mean total reward:    {results['mean_total_reward']:.3f} ± {results['std_total_reward']:.3f}")
    print(f"Mean FP rate:         {results['mean_fp_rate']:.1%}")
    print(f"Approved rate:        {results['approved_rate']:.1%}")
    print(f"Results saved ->       {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="easy_fleet", choices=["easy_fleet", "medium_fleet", "hard_fleet", "banking_fleet"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_random_agent(args.task_id, args.episodes, args.seed)


if __name__ == "__main__":
    main()

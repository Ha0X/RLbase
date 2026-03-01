"""
REINFORCE (Policy Gradient) - Simple entrypoint for CartPole-v1.

The full implementation with CLI options is in `scripts/basics/reinforce/`.
This entrypoint provides a simple default run.
"""
import sys
from pathlib import Path

# Add reinforce/ to path so we can import from src
reinforce_dir = Path(__file__).parent / "reinforce"
sys.path.insert(0, str(reinforce_dir))

from main import parse_args
from src.train import train
from src.utils import seed_everything


def main():
    """Run REINFORCE on CartPole-v1 with default settings."""
    args = parse_args()
    args.env = "CartPole-v1"
    args.total_steps = 100_000
    args.lr = 3e-3
    args.gamma = 0.99
    args.baseline = False  # Pure REINFORCE
    args.seed = 42
    seed_everything(args.seed)
    train(args)


if __name__ == "__main__":
    main()


"""StressSignal: wearable stress detection on the Stress-Predict dataset."""


def main() -> None:
    from .train import main as train

    train()

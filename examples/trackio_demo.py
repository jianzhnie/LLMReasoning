import math
import random
import time
from typing import Any, Dict, List

# The `trackio` library is aliased to `wandb` for a familiar interface,
# but the original code had an incorrect import statement.
# Correcting it to use `trackio` and aliasing it to `wandb`.
import trackio as wandb  # Assuming 'wandb' is the desired alias for trackio


# --- Data Generation Functions ---
def generate_loss_curve(
    epoch: int,
    num_epochs: int,
    base_loss: float = 2.5,
    min_loss: float = 0.1,
) -> float:
    """
    Generate a realistic, noisy loss curve that decreases over time.

    The function simulates a machine learning training loss curve, which
    generally starts high and decreases as training progresses. It uses
    an exponential decay curve and adds Gaussian noise that also
    diminishes over time, mimicking real-world training behavior.

    Args:
        epoch: The current epoch number, starting from 0.
        num_epochs: The total number of epochs for the simulation.
        base_loss: The starting loss value. Defaults to 2.5.
        min_loss: The minimum theoretical loss value the curve approaches.
                  Defaults to 0.1.

    Returns:
        The simulated loss value for the given epoch.
    """
    # Calculate the normalized progress of the training.
    progress: float = epoch / num_epochs
    # Exponential decay curve to simulate decreasing loss.
    base_curve: float = base_loss * math.exp(-3 * progress) + min_loss

    # Add Gaussian noise that scales down with progress.
    noise_scale: float = 0.3 * (1 - progress * 0.7)
    noise: float = random.gauss(0, noise_scale)

    # Ensure the loss does not drop to an unrealistic value.
    simulated_loss: float = base_curve + noise
    return max(min_loss * 0.5, simulated_loss)


def generate_accuracy_curve(
    epoch: int,
    num_epochs: int,
    max_acc: float = 0.95,
    min_acc: float = 0.1,
) -> float:
    """
    Generate a realistic, noisy accuracy curve that increases over time.

    This function models an accuracy curve using a sigmoid function, which
    simulates the characteristic S-shape of training accuracy curves. It also
    adds decreasing Gaussian noise to reflect the stabilization of metrics.

    Args:
        epoch: The current epoch number.
        num_epochs: The total number of epochs.
        max_acc: The maximum achievable accuracy. Defaults to 0.95.
        min_acc: The initial accuracy. Defaults to 0.1.

    Returns:
        The simulated accuracy value for the given epoch.
    """
    # Calculate the normalized progress.
    progress: float = epoch / num_epochs
    # Sigmoid curve to simulate accuracy increase.
    base_curve: float = max_acc / (1 + math.exp(-6 *
                                                (progress - 0.5))) + min_acc

    # Add Gaussian noise that scales down with progress.
    noise_scale: float = 0.08 * (1 - progress * 0.5)
    noise: float = random.gauss(0, noise_scale)

    # Clamp the accuracy to a valid range [0, max_acc].
    simulated_accuracy: float = base_curve + noise
    return max(0.0, min(max_acc, simulated_accuracy))


# --- Simulation Function ---
def run_single_simulation(epochs: int) -> None:
    """
    Simulates a single machine learning training run.

    This function iterates through a set number of epochs, generates
    realistic training and validation metrics for each epoch, and logs
    them to `wandb`. It also introduces a small, random fluctuation to
    add more realism to the validation metrics.

    Args:
        epochs: The total number of epochs to simulate.
    """
    # Simulate training for the specified number of epochs.
    for epoch in range(epochs):
        # Generate dynamic training and validation metrics using random parameters.
        train_loss: float = generate_loss_curve(
            epoch,
            epochs,
            base_loss=random.uniform(2.5, 3.5),
            min_loss=random.uniform(0.05, 0.15),
        )
        val_loss: float = generate_loss_curve(
            epoch,
            epochs,
            base_loss=random.uniform(2.5, 3.5),
            min_loss=random.uniform(0.05, 0.15),
        )
        train_accuracy: float = generate_accuracy_curve(
            epoch,
            epochs,
            max_acc=random.uniform(0.7, 0.9),
            min_acc=random.uniform(0.1, 0.3),
        )
        val_accuracy: float = generate_accuracy_curve(
            epoch,
            epochs,
            max_acc=random.uniform(0.7, 0.9),
            min_acc=random.uniform(0.1, 0.3),
        )

        # Introduce a random "hiccup" or fluctuation in validation metrics.
        if epoch > 2 and random.random() < 0.3:
            val_loss *= 1.1
            val_accuracy *= 0.95

        # Prepare metrics for logging with appropriate rounding.
        metrics_to_log: Dict[str, float] = {
            'train/loss': round(train_loss, 4),
            'train/accuracy': round(train_accuracy, 4),
            'train/rewards/reward1': random.random(),
            'train/rewards/reward2': random.random(),
            'val/loss': round(val_loss, 4),
            'val/accuracy': round(val_accuracy, 4),
        }

        # Log metrics to the current wandb run.
        wandb.log(metrics_to_log)


# --- Main Execution Block ---
def main() -> None:
    """
    Orchestrates multiple simulated training runs for a project.

    This function configures and initializes multiple `wandb` runs, each
    simulating a distinct machine learning training session. It sets
    up a unique project ID and logs key configuration parameters for
    each run.
    """
    # --- Constants and Configuration ---
    # Total number of epochs for each simulation run.
    epochs: int = 20

    project_id: str = 'trackio-demo'
    run_id: str = 'test-1'
    # Initialize a new wandb run with a unique name and configuration.
    # It's good practice to set a descriptive project name, a unique run name,
    # and log key hyperparameters in the config dictionary.
    wandb.init(
        project=f'{project_id}',
        name=f'{run_id}',
        config={
            'epochs': epochs,
            'learning_rate': 0.001,
            'batch_size': 32,
        },
        # Note: space_id and dataset_id are specific to trackio, not standard wandb.
        space_id='jianzhnie/LLMReasoning',
        dataset_id='LLMReasoning-{project_id}',
    )
    # Run the simulation for the specified number of epochs.
    run_single_simulation(epochs)

    # Finalize the current wandb run to close it properly.
    wandb.finish()
    print('Completed simulation run.')


if __name__ == '__main__':
    main()

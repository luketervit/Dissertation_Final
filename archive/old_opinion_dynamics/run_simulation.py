"""
Main entry point for running the Historical Replay ABM simulation.

Usage:
    python sim/run_simulation.py
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.model import HistoricalReplayModel


def main():
    """Run the simulation and save results."""
    print("="*80)
    print("HISTORICAL REPLAY ABM - BOUNDED CONFIDENCE + BACKFIRE EFFECTS")
    print("="*80)

    # Initialize model
    print("\nInitializing model...")
    model = HistoricalReplayModel(config_path='config/thread_config.yaml')

    # Run simulation
    model.run_simulation()

    # Get results
    print("\nCollecting results...")
    model_data, agent_data = model.get_results()
    agent_summaries = model.get_agent_summaries()

    # Save results
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    model_data.to_csv(output_dir / 'simulation_model_data.csv')
    agent_data.to_csv(output_dir / 'simulation_agent_data.csv')
    agent_summaries.to_csv(output_dir / 'simulation_agent_summaries.csv', index=False)

    print(f"\n✓ Results saved:")
    print(f"  {output_dir / 'simulation_model_data.csv'}")
    print(f"  {output_dir / 'simulation_agent_data.csv'}")
    print(f"  {output_dir / 'simulation_agent_summaries.csv'}")

    # Summary statistics
    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)

    print(f"\nOpinion Shifts:")
    for label in ['Left', 'Right', 'Center']:
        agents = agent_summaries[agent_summaries['political_label'] == label]
        if len(agents) > 0:
            mean_shift = agents['opinion_shift'].mean()
            print(f"  {label:8s}: Mean shift = {mean_shift:+.4f} (n={len(agents)})")

    print(f"\nOverall:")
    print(f"  Mean opinion shift: {agent_summaries['opinion_shift'].mean():+.4f}")
    print(f"  Opinion shift std: {agent_summaries['opinion_shift'].std():.4f}")
    print(f"  Max positive shift: {agent_summaries['opinion_shift'].max():+.4f}")
    print(f"  Max negative shift: {agent_summaries['opinion_shift'].min():+.4f}")

    print(f"\nExposure:")
    print(f"  Mean tweets viewed: {agent_summaries['tweets_viewed'].mean():.1f}")
    print(f"  Total exposures: {agent_summaries['tweets_viewed'].sum()}")

    # Plot opinion trajectories
    print("\nGenerating plots...")
    plot_results(model_data, agent_summaries)

    print("\n" + "="*80)
    print("SIMULATION COMPLETE!")
    print("="*80)


def plot_results(model_data, agent_summaries):
    """Generate visualization plots."""
    output_dir = Path('output')

    # 1. Opinion drift over time
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Mean opinion over time
    axes[0, 0].plot(model_data['time'], model_data['mean_opinion'], 'b-', label='All agents')
    axes[0, 0].plot(model_data['time'], model_data['left_opinion'], 'r--', label='Left agents')
    axes[0, 0].plot(model_data['time'], model_data['right_opinion'], 'g--', label='Right agents')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Mean Opinion')
    axes[0, 0].set_title('Opinion Drift Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Opinion variance over time
    axes[0, 1].plot(model_data['time'], model_data['opinion_variance'], 'purple')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Opinion Variance')
    axes[0, 1].set_title('Opinion Polarization Over Time')
    axes[0, 1].grid(True, alpha=0.3)

    # Opinion shift distribution
    axes[1, 0].hist(agent_summaries['opinion_shift'], bins=30, edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Opinion Shift')
    axes[1, 0].set_ylabel('Number of Agents')
    axes[1, 0].set_title('Distribution of Opinion Shifts')
    axes[1, 0].grid(True, alpha=0.3)

    # Opinion shift by political label
    for label in ['Left', 'Right', 'Center']:
        data = agent_summaries[agent_summaries['political_label'] == label]['opinion_shift']
        if len(data) > 0:
            axes[1, 1].hist(data, bins=20, alpha=0.5, label=label)
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Opinion Shift')
    axes[1, 1].set_ylabel('Number of Agents')
    axes[1, 1].set_title('Opinion Shifts by Political Label')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'simulation_results.png', dpi=150)
    print(f"  ✓ Saved: {output_dir / 'simulation_results.png'}")

    plt.close()


if __name__ == '__main__':
    main()


import os
import sys
import yaml
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Import simulation directly
try:
    from sim.thread_simulation import ThreadModel
except ImportError:
    logging.error("Could not import sim.thread_simulation. Run from project root?")
    sys.exit(1)

def run_simulation_run(config_path, output_dir):
    """Run a single simulation."""
    logging.info(f"Starting simulation run with config: {config_path}")
    
    try:
        model = ThreadModel(config_path=str(config_path))
        
        # Run simulation for steps
        steps = model.config['simulation']['max_rounds'] * 20 # Approximate steps
        logging.info(f"Running for {steps} steps (approx)")
        
        # Simple run loop
        for i in range(steps):
            model.step()
            # Stop if max rounds reached or thread too deep
            if model.current_round >= model.config['simulation']['max_rounds']:
                break
                
        # Save output
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save thread history
        history = [p for p in model.thread_history]
        with open(output_dir / 'thread_history.json', 'w') as f:
            json.dump(history, f, indent=2)
            
        logging.info(f"Simulation complete. Saved to {output_dir}")
        return history
        
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_results(history, label):
    """Calculate basic metrics for a run."""
    if not history:
        return {}
        
    # Filter for AI replies (exclude root)
    replies = [p for p in history if p['post_id'] != 0]
    if not replies:
        return {'n_replies': 0}
        
    aggs = [p.get('aggression', 0) for p in replies]
    pols = [p.get('political_label') for p in replies]
    
    # Negativity % (Aggression > 0.5 is a proxy for Toxic)
    # Using 0.5 assuming hate+offensive sum
    neg_count = sum(1 for a in aggs if a > 0.5)
    neg_pct = (neg_count / len(replies)) * 100
    
    mean_agg = np.mean(aggs)
    
    # Political Dist
    counts = pd.Series(pols).value_counts(normalize=True)
    
    return {
        'label': label,
        'n_replies': len(replies),
        'mean_aggression': mean_agg,
        'negativity_pct': neg_pct,
        'left_pct': counts.get('Left', 0),
        'right_pct': counts.get('Right', 0)
    }

def main():
    # Target Thread
    THREAD_ID = 'thread_001'
    RECON_DIR = PROJECT_ROOT / 'batch_simulations_reconstructed' / THREAD_ID
    BASE_OUTPUT_DIR = PROJECT_ROOT / 'ablation_output'
    
    if not RECON_DIR.exists():
        logging.error(f"Thread directory not found: {RECON_DIR}")
        return

    # Prepare Configs
    # We need to construct a config that points to the correct files in RECON_DIR
    
    base_config = {
        'target_tweet_id': '1801016461601001478', # Hardcoded from existing, or load from metadata
        'paths': {
            'thread_metadata': str(RECON_DIR / 'thread_metadata.json'),
            'agents_for_thread': str(RECON_DIR / 'agents_for_thread.csv'),
        },
        'llm': {
            'provider': 'ollama',
            'model': 'dolphin-llama3:8b',
            'temperature': 0.9,
            'max_tokens': 150,
        },
        'simulation': {
            'max_rounds': 10,
            'thread_num': 1
        }
    }
    
    # 1. Zero-Shot Run
    logging.info("--- Preparing Zero-Shot Run ---")
    config_0shot = base_config.copy()
    config_0shot['simulation'] = base_config['simulation'].copy()
    config_0shot['simulation']['use_few_shot'] = False
    
    path_0shot = PROJECT_ROOT / 'config_ablation_0shot.yaml'
    with open(path_0shot, 'w') as f:
        yaml.dump(config_0shot, f)
        
    out_0shot = BASE_OUTPUT_DIR / '0_shot' / THREAD_ID
    
    # 2. Few-Shot Run
    logging.info("--- Preparing Few-Shot Run ---")
    config_fewshot = base_config.copy()
    config_fewshot['simulation'] = base_config['simulation'].copy()
    config_fewshot['simulation']['use_few_shot'] = True 
    
    path_fewshot = PROJECT_ROOT / 'config_ablation_fewshot.yaml'
    with open(path_fewshot, 'w') as f:
        yaml.dump(config_fewshot, f)
        
    out_fewshot = BASE_OUTPUT_DIR / 'few_shot' / THREAD_ID

    # Execution
    print("\n" + "="*50)
    print("RUNNING ZERO-SHOT SIMULATION")
    print("="*50)
    res_0shot = run_simulation_run(path_0shot, out_0shot)
    
    print("\n" + "="*50)
    print("RUNNING FEW-SHOT SIMULATION")
    print("="*50)
    res_fewshot = run_simulation_run(path_fewshot, out_fewshot)
    
    # Analysis
    stats_0 = analyze_results(res_0shot, "Zero-Shot")
    stats_few = analyze_results(res_fewshot, "Few-Shot")
    
    print("\n" + "="*50)
    print("ABLATION RESULTS: NEGATIVITY GAP")
    print("="*50)
    
    comparison = pd.DataFrame([stats_0, stats_few])
    print(comparison.to_string(index=False))
    
    # Simple interpretation
    if stats_few.get('negativity_pct', 0) > stats_0.get('negativity_pct', 0):
        print("\nCONCLUSION: Few-Shot generates MORE hostility than Zero-Shot.")
        print(f"Gap: +{stats_few['negativity_pct'] - stats_0['negativity_pct']:.2f}%")
    else:
        print("\nCONCLUSION: Zero-Shot generates MORE/EQUAL hostility (Surprising).")

    # Clean up configs
    if path_0shot.exists(): path_0shot.unlink()
    if path_fewshot.exists(): path_fewshot.unlink()

if __name__ == "__main__":
    main()

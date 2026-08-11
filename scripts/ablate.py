import yaml
import argparse
from scripts.train import train

def run_ablation(config_path: str, mechanic_to_disable: str):
    print(f"Running Ablation: NO {mechanic_to_disable.upper()}")
    # In a full implementation, we would patch the configuration and the 
    # environment logic to physically block discovery or exchange.
    # For this prototype, we'll demonstrate the framework hook.
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    config['logging']['results_dir'] = f"results_no_{mechanic_to_disable}"
    
    # Save temp config
    temp_path = f"configs/temp_ablate_{mechanic_to_disable}.yaml"
    with open(temp_path, 'w') as f:
        yaml.dump(config, f)
        
    # We would theoretically pass a flag to RAIEnv to zero out action masks for 
    # exchange or explore here.
    
    train(temp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--disable", type=str, required=True, choices=["exchange", "discovery"])
    args = parser.parse_args()
    
    run_ablation(args.config, args.disable)

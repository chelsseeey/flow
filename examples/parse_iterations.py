import json
import pandas as pd
import os

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def parse_training_results(trial_name):
    # Ray results base path
    base_path = "/root/ray_results/figure8_with_lights"
    
    # Find trial folder
    trial_path = os.path.join(base_path, trial_name)
    if not os.path.exists(trial_path):
        raise FileNotFoundError(f"Trial folder not found: {trial_path}")
    
    # Result.json path
    json_file = os.path.join(trial_path, 'result.json')
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"No result.json found in {trial_path}")
    
    iterations = []
    with open(json_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            flat_data = flatten_dict(data)
            iterations.append(flat_data)
    
    df = pd.DataFrame(iterations)
    
    # Create csv_results directory in base path
    results_dir = os.path.join(base_path, 'csv_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save CSV with trial name
    csv_path = os.path.join(results_dir, f'{trial_name}_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Display first few rows
    print("\nPreview of results:")
    print(df.head())
    
    # Open CSV in VS Code
    os.system(f"code {csv_path}")
    
    return df

if __name__ == "__main__":
    trial_name = input("Enter trial name : ")
    parse_training_results(trial_name)
import jsonlines
import subprocess
import sys
import os

# config
jsonl_file_path = 'data/harmful_behaviors.jsonl'
attack_model = 'vicuna'
judge_model = 'gpt-3.5-turbo-0125'
script_path = 'main.py'
log_dir = 'logs'
target_model = 'gpt-3.5-turbo-0125'

os.makedirs(log_dir, exist_ok=True)

def run_experiment(goal, target_str):
    log_file_path = os.path.join(log_dir, f"{goal}.log")
    command = [
        'python3', script_path,
        '--attack-model', attack_model,
        '--target-model', target_model,
        '--judge-model', judge_model,
        '--goal', f'"{goal}"',
        '--target-str', f'"{target_str}"'
    ]
    
    print(f"Running command: {' '.join(command)}")
    with open(log_file_path, 'w') as log_file:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        log_file.write(f"Command: {' '.join(command)}\n")
        log_file.write(f"STDOUT:\n{stdout}\n")
        log_file.write(f"STDERR:\n{stderr}\n")
        print(f"Logs saved to: {log_file_path}")
        print("-" * 40)

def main():
    with jsonlines.open(jsonl_file_path) as reader:
        for obj in reader:
            goal = obj.get('query')
            target_str = obj.get('target')
            
            if goal and target_str:
                safe_goal = goal.replace('/', '_').replace('\\', '_')
                run_experiment(safe_goal, target_str)
            else:
                print(f"Error: Missing 'query' or 'target' in the JSONL file.")

if __name__ == "__main__":
    main()

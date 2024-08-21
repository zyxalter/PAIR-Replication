# PAIR Experiment

## Overview

**PAIR (Prompt Automatic Iterative Refinement)**: For the baseline of semantic-level black-box jailbreaks, we utilized the representative method PAIR (Chao et al., 2023) to evaluate its attack performance and stealth capabilities. PAIR employs an attacker Large Language Model (LLM) to automatically generate jailbreak prompts for another target LLM without human intervention. The attacker LLM repeatedly queries the target LLM to iteratively update and optimize candidate jailbreaks.

To achieve optimal results, we set **N=60 streams**, each with a maximum depth of **K=3**, and used the same additional parameters as outlined in [Chao et al., 2023]. Detailed system prompts and generation parameters can be found in **Section A**.

## Getting Started

We provide the necessary environment for running the experiments, listed in `requirements.txt`.
Ensure that your API key is stored in the `OPENAI_API_KEY` environment variable:
  ```bash
  export OPENAI_API_KEY=[YOUR_API_KEY_HERE]
  ```
  Modify the `config.py` file to set the correct local paths for Vicuna or Llama models.

## Running the Experiment
To run the experiment, execute the following command:
  ```bash
    python3 main.py --attack-model [ATTACK MODEL] --target-model [TARGET MODEL] --judge-model [JUDGE MODEL] --goal [GOAL STRING] --target-str [TARGET STRING]
 ```
By default, the experiment uses `--n-streams 60` and `--n-iterations 3`.
**Reproducing Results from the Paper：** If you wish to run the experiment with the exact settings used in the paper, you can execute:
```bash
python3 run_experiment.py
 ```
 This command will conduct batch experiments on 50 harmful behaviors listed in `data/harmful_behaviors.jsonl`, using the same settings as described in the paper. Upon completion, the final generated attack prompts and iteration history will be saved in the results folder.
 
## Analyzing Results with ASR and LlamaGuard
 To analyze the results, you can run:
 ```bash
python3 check_ASR_and_LlamaGuard.py --file_path [FILE PATH]
 ```
 Here, `[FILE PATH]` refers to the JSON file you wish to analyze for ASR statistics. A sample file is provided at `sample_result/final_prompt.json`.

If you have already run python run_experiment.py, you can directly execute:
 ```bash
python3 check_ASR_and_LlamaGuard.py
 ```
 This will generate and display statistics regarding the effectiveness of the attacks from the experiment.
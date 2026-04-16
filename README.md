# Mitigating Social Loafing in CS Group Projects Using Concordia

# To Run:
## 1. Add your Gemini API Key

`export GEMINI_API_KEY="add_your_api_key_here"`

## 2. Generate the Trait Pool

`python generate_trait_pool.py`

The trait pool will be outputted into `./configs/trait_pool.yaml`

## 3. Run the model
You can change the --n_trials (number of trials;default is 100), --n_agents (number of agents; default is 5), --max_steps (max steps; default is 20), --max_workers (max workers; default is 8), --base_seed (base seed; default is 42), and --output (output file; default is results/simulation_results.jsonl)
```
python simulations/run_parallel.py \
  --n_trials 10 \
  --max_workers 10 \
  --max_steps 10 \
  --output results/simulation_results_1.jsonl
  ```
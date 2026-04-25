# Mitigating Social Loafing in CS Group Projects Using Concordia

# To Run:
## 1. Add your Gemini API Key

`export GEMINI_API_KEY="add_your_api_key_here"`

## 2. Generate the Trait Pool

`python generate_trait_pool.py`

The trait pool will be outputted into `./configs/trait_pool.yaml`

## 3. Generate the 100 groups of 5 random agents

`python configs/generate_groups.py`

The groups will be outputted into `simulation_groups.jsonl`

## 4. Run the Concordia simulations (parallelized using Ray)
To run all of the tests (control + 6 interventions), run:

`python simulations/run_parallel_ray.py --condition all`

## 5. Rate the loafing for each agent in each group for every trial using Mulvey and Klein (1998)’s Perceived Social Loafing Questionnaire

Run this for the control and change the file names for the interventions:

`python results/perceived_loafing.py results/pl_outputs/control_pl_score.csv --output-csv output_files/control_output.csv`

## 6. Run the Benjamini Hochberg procedure

Analyze the results:

`python results/benjamini_hochberg.py`
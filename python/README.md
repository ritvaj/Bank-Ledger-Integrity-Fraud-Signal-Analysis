
# Python Pipeline

This folder contains scripts for the full fraud analytics workflow.

## Structure
etl/
  • cleaning.py
  • feature_engineering_mismatch.py
  • feature_engineering_behavioral.py

analysis/
  • plots.py
  • ab_testing.py

## Run Order (run from repo root)
1. python/etl/cleaning.py
2. python/etl/feature_engineering_mismatch.py
3. python/etl/feature_engineering_behavioral.py
4. python/analysis/plots.py
5. python/analysis/ab_testing.py

## Required Input File
Place your dataset (PaySim or equivalent) at the project root as:

`Transactions_200kII.csv`

or modify the script paths to match your file name.

## Outputs
• Figures → outputs/figures/  
• Summary tables → outputs/tables/  
• A/B test metrics + high-risk accounts → outputs/tables/

## Requirements
pandas, numpy, seaborn, matplotlib, scikit-learn

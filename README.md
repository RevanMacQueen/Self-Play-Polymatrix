# Code For Guarantees for Self-Play in Multiplayer Games via Polymatrix Decomposability

This is the Repo for Guarantees for Self-Play in Multiplayer Games via Polymatrix Decomposability. The paper is available here. 

## Installation

## Recreating Results
Recreating the results from the paper involves two steps:
1. Generate strategies
2. Use SGDecompose to find subgame stable CSP games

### Recreating strategies. 
The python script run_cfr.py will run cfr/cfr+ on a game specified via command line arguments. Please refer to run_cfr.py for all available arguments. We randomly initialize the regrets of cfr/cfr+, where the initialization depends on the seed argument. 


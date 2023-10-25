# Code For Guarantees for Self-Play in Multiplayer Games via Polymatrix Decomposability

This is the repo for [Guarantees for Self-Play in Multiplayer Games via Polymatrix Decomposability](https://arxiv.org/abs/2310.11518).  

## Installation

We use Python 3.9. Requirements are included in `requirements.txt`. 

This codebase is based on the [OpenSpiel](https://github.com/google-deepmind/open_spiel) framework. We use a modified version of OpenSpiel, which allows for CFR/CFR+ to be randomly initialized with the python API. Begin by installing this framework [here](https://github.com/RevanMacQueen/open_spiel). You will also need to install [Gurobi](https://www.gurobi.com/). 
 
##  Recreating Results
Recreating the results from the paper involves two steps:
1. Generate strategies
2. Running SGDecompose 

### 1. Generating Strategies
The python script `run_cfr.py` will run CFR/CFR+ on a game specified via command line arguments. Please refer to `run_cfr.py` for all available arguments. We randomly initialize the regrets of CFR/CFR+, where the initialization depends on the seed argument. 

Files have been provided with all the necessary runs for our results. These are contained in:
* `run_CFR_leduc.txt`
* `run_CFR_plus_leduc.txt`
* `run_CFR_hanabi.txt`

### 2. Running SGDecompose
Once the CFR strategies have been generated, use `run_sgdecompose.py` to produce an approximate SS-CSP game.  Use the command line arguments to specify the hyperparameters and save directory of the CFR strategies. 

Note that we represent strategies as a vector, where there is an entry for each terminal history, which gives a player’s contribution to reaching that history. Thus, multiplying all player’s strategies together will give a probability distribution	over terminals. This trick makes computing expected utilities in large games like Leduc poker much more efficient. 

## Example Using Kuhn Poker
Let’s walk through an example using an even smaller version of poker, Kuhn poker. The commands to generate 30 strategy profiles with CFR are in `run_cfr_kuhn.txt`; alternatively, run `python3 run_cfr_kuhn.py`. This will save strategies (in vector form) in `cfr/cfr_kuhn`.  For example, `cfr/cfr_kuhn/0/strats/1.npy`is the strategy for player 1 in run 0.  The values of epsilon for approximate Nash equilibria are also saved; for example, `cfr/cfr_kuhn/1/epsilons/300.npy`is a numpy array of deviation incentives for each player after CFR has been trained for 300 iterations for run 1. 

Now, let’s use SGDecompose to compute a CSP game. Running 
```
python3 run_sgdecompose.py --game kuhn_poker --strat_dir cfr/cfr_kuhn  --save_dir results/kuhn --epochs 100
```
will run SGDecompose for 100 epochs using these strategies. This should take fewer than 5 minutes. 

At the end, we’re left with an output file `results/kuhn/log.json`. Inspecting this file, we see under “gamma” a value of gamma for each epoch. The final value is ~0.00025; meaning that SGDecompose found a  CSP game that was 0.00025-subgame stable in the neighborhood of the strategies learned by CFR. Similarly for delta, we see that Kuhn poker is ~0.0042-CSP in the neighborhood of the CFR strategies. The vulnerability with respect to this set is ~0.006, and the value implied by the bounds in the paper is ~0.009. 

## Overview of Files
* `results/` contains the specific results we used in the paper; these are log files from running SGDecompose.  There is a subdirectory for each run, giving 30 in total
* `analysis.ipynb` will generate the plots from the paper using these results. 
* `sgdecompose/` contains the implementation of  SGDecompose. This directory contains the following files: 
  * `sequence.py`, a modified version of OpenSpeil’s sequence-form linear programming code. The code was adapted for use with more than 3 players. The most important function  is `construct_vars()`, which returns the necessary information for contracting LPs. 
  * `sgdecompose_utils.py`, utilities for `sgdecompose.py`.
  * `sgdecompose.py`implements SGDecompose from the paper. 
  * `subgame_solver.py` contains a helper class for computing best responses in subgames. 
  * `terminals.py` contains utilities related to terminals, such as getting the returns/utility and reach probabilities at each terminal.
* `policy.py` a modified version of OpenSpiel’s TabularPolicy class that allows for random initializations. 
* The following files are used for generating CFR/CFR+ learned strategies:
  * `run_CFR_leduc.txt`
  * `run_CFR_plus_leduc.txt`
  * `run_CFR_hanabi.txt`
* `run_cfr.py` runs CFR/CFR+.
* `run_sgdecompose.py` runs SGDecompose. 
* We also include two files for the example using Kuhn poker. These are:
  * `run_cfr_kuhn.py`
  * `run_cfr_kuhn.txt`



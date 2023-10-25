import jax.numpy as np
import numpy as onp
import pyspiel
import argparse

from tqdm import tqdm
from pathlib import Path

import sgdecompose.terminals as terminals

from run_cfr import build_tiny_hanabi
from sgdecompose.sgdecompose import sgdecompose
from policy import TabularPolicy


def get_args():
    """
    This function will extract the arguments from the command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='leduc_poker', type=str, nargs='?')
    parser.add_argument('--num_players', default=3, type=int, nargs='?')
    parser.add_argument('--epochs', default=1000, type=int, nargs='?')
    parser.add_argument('--batch_size', default=30, type=int, nargs='?')
    parser.add_argument('--lr', default=1e-3, type=float, nargs='?')
    parser.add_argument('--lambda', default=1/2, type=float, nargs='?')
    parser.add_argument('--seed', default=1, type=int, nargs='?')
    parser.add_argument('--start', default=0, type=int, nargs='?') # start of the strategies to load
    parser.add_argument('--num_strats', default=30, type=int, nargs='?') # number of strategies to load
    parser.add_argument('--strat_dir', default='cfr/cfr_plus_leduc', type=str, nargs='?') # root of saved strategies 
    parser.add_argument('--random', default=False, type=bool, nargs='?')
    parser.add_argument('--save_dir', default='results/poly_leduc_cfr_plus', type=str, nargs='?')
    parser.add_argument('--class', default='zsp', choices=['zsp', 'csp'])
    return vars(parser.parse_args())


def make_random_strats(game, num_players, num_strats):
    strats = [[] for _ in range(num_players)]
    joint_policies = []
    for p_i in range(num_players):
        for _ in range(num_strats):
            jp = TabularPolicy(game, blueprint=True)
            joint_policies.append(jp)
            reach_probs = terminals.player_terminal_reach_probs(game, jp, p_i)
            reach_probs = reach_probs
            strats[p_i].append(np.expand_dims(reach_probs, 0))
        strats[p_i] = np.concatenate(strats[p_i], axis=0)
    return strats


def main():
    ARGS = get_args()
    GAME = ARGS['game']
    NUM_PLAYERS = ARGS['num_players']
    NUM_STRATS = ARGS['num_strats']
    STRAT_DIR = ARGS['strat_dir']
    SEED = ARGS['seed']
    START = ARGS['start'] 
    SAVE_DIR = ARGS['save_dir']

    assert NUM_PLAYERS == 3 # everything is hard coded for 3 players for now. 

    onp.random.seed(SEED)
    
    # generate game
    if GAME == 'tiny_hanabi':
        game = build_tiny_hanabi()
    else:
        game = pyspiel.load_game(GAME, {"players": NUM_PLAYERS})

    # load strategies 
    if ARGS['random']:
        strats = make_random_strats(game, NUM_PLAYERS, NUM_STRATS)
    else:
        strats = []
        for i in range(NUM_PLAYERS):
            player_strats = []
            for run in tqdm(range(START, START+ NUM_STRATS)):
                s = np.load(Path(STRAT_DIR)/str(run)/'strats'/'{}.npy'.format(i))        
                player_strats.append(s)
            strats.append(np.array(player_strats))

    hyperparams = {
        'epochs' : ARGS['epochs'],
        'lr' : ARGS['lr'],
        'batch_size' : ARGS['batch_size'],
        'lambda' : ARGS['lambda'],
        'class' : ARGS['class']
    }

    game_info = {
        'game' : game,
        'num_players' : NUM_PLAYERS,
        'num_strats' : NUM_STRATS
    }

    sgdecompose(strats, game_info, hyperparams, seed=SEED, save_dir=SAVE_DIR)

if __name__ == '__main__':
    main()
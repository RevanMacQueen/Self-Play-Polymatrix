import argparse
import numpy as np
import pyspiel
from tqdm import tqdm
from pathlib import Path
import pickle
from open_spiel.python.algorithms.exploitability import nash_conv
import sgdecompose.terminals as terminals

def build_tiny_hanabi():
    num_chance = 2
    num_actions = 2
    num_players = 3
    payoff = ''
    n = 0
    for c_1 in range(num_chance): # the chance action
        for _ in range(num_chance): # the chance action
            for _ in range(num_chance): # the chance action
                for a_1 in range(num_actions):  # p1 action
                    for a_2 in range(num_actions): # p2 action
                        for a_3 in range(num_actions): # p3 action
                            if c_1 == a_2 == a_3:
                                payoff += '1;'
                            else:
                                payoff += '0;'
                            n += 1
                        
    payoff = payoff[:-1]
    game = pyspiel.load_game('tiny_hanabi',  {"num_players" : num_players,  
                                          'num_chance' : num_chance, 
                                          'num_actions' : num_actions,  
                                          "payoff" : payoff})
    return game


def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, nargs='?')
    parser.add_argument('--iterations', default=1000, type=int, nargs='?')
    parser.add_argument('--game', default='leduc_poker', type=str, nargs='?')
    parser.add_argument('--num_players', default=3, type=int, nargs='?')
    parser.add_argument('--save_dir', default='cfr_test', type=str, nargs='?')
    parser.add_argument('--save_freq', default=1000, type=int, nargs='?')
    parser.add_argument('--eval_freq', default=100, type=int, nargs='?')
    parser.add_argument('--alg', default='cfr_plus', choices=['cfr', 'cfr_plus'], type=str, nargs='?')
    return vars(parser.parse_args())


def save_policy(dir, solver, policy):
    # save a policy as a dictionary of infoset, action pairs
    dir.mkdir(parents=True, exist_ok=True)
    pickle.dump(policy, open(dir/'policy.pkl', 'wb'))
    pickle.dump(solver, open(dir/'cfr.pkl', 'wb'))

if __name__ == '__main__':
    ARGS = get_args()
    SEED = ARGS['seed']
    NUM_PLAYERS = ARGS['num_players']
    ITERATIONS =  ARGS['iterations']
    GAME = ARGS['game']
    SAVE_DIR = Path(ARGS['save_dir'])
    SAVE_FREQ = ARGS['save_freq'] 
    EVAL_FREQ = ARGS['eval_freq']

    if GAME == 'tiny_hanabi':
        game = build_tiny_hanabi()
    else:
        game = pyspiel.load_game(GAME, {"players": NUM_PLAYERS})

    np.random.seed(SEED)
   
    start_itr = 0
    (SAVE_DIR/'epsilons').mkdir(parents=True, exist_ok=True) # where to save the epsilons
    (SAVE_DIR/'strats').mkdir(parents=True, exist_ok=True) # where to save the epsilons

    if ARGS['alg'] == 'cfr_plus': 
        cfr = pyspiel.CFRPlusSolver(game, SEED, True) # second arg is to use random regrets initially
    elif ARGS['alg'] == 'cfr': 
        cfr = pyspiel.CFRSolver(game, SEED, True) # second arg is to use random regrets initially

    for itr in tqdm(range(ITERATIONS)): 
        cfr.evaluate_and_update_policy()
        if itr %EVAL_FREQ == 0:
            cfr_policy = cfr.average_policy() # NOTE: cfr_policy is the marginal strategy profile of a CCE 
            epsilons = nash_conv(game, cfr_policy , use_cpp_br=True, return_only_nash_conv=False).player_improvements
            np.save(SAVE_DIR/'epsilons'/'{}.npy'.format(itr), epsilons)

    cfr_policy = cfr.average_policy() 
    epsilons = nash_conv(game, cfr_policy , use_cpp_br=True, return_only_nash_conv=False).player_improvements
    np.save(SAVE_DIR/'epsilons'/'{}.npy'.format(itr), epsilons)
    
    # save each players strategy as a "distribution" over terminals, giving their contribution to reaching that terminal. 
    policy = cfr.average_policy()
    for p in range(NUM_PLAYERS):
        np.save(
            SAVE_DIR/'strats'/'{}.npy'.format(p),
            terminals.player_terminal_reach(game, policy, p)
        )
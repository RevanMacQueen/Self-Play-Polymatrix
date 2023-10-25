import jax.numpy as np
import jax
import json
import time

from itertools import product
from tqdm import tqdm
from pathlib import Path

import sgdecompose.terminals as terminals
import sgdecompose.sgdecompose_utils as utils

get_util = None

def sgdecompose(strats, game_info, hyperparams, seed=0, save_dir='results'):
    '''
    The main funciton for SGDecompose. This function will compute a CSP game that is delta-CSP in the neighborhood of all possible match-ups of strats and gamma-subgame stable in the neighbourhood of strats. It will try and find small values of gamma and delta. 

    args : 
        strats : the strategies learned in self-play. 
        game_info : dictionary with information about the game
        hyperparams : hyperparameters for SGDecompose
        seed : random seed
        save_dir : directory to save results
    '''
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # information about the game
    num_players = game_info['num_players']
    game = game_info['game']
    num_strats = game_info['num_strats']
    chance_reach = terminals.chance_terminal_reach_probs(game) # chance's reach prob
    u_game = terminals.get_terminal_returns(game)  * chance_reach  # we include chance reach in terminal returns for ease
    num_terminals = len(chance_reach)

    # Cast list of strategies into a numpy array, this will have shape (num_players, num_strategies, num_terminals).
    strats = np.array(strats)


    # set jax key
    key = jax.random.PRNGKey(seed)

    # get hyperparameters
    game_class =  hyperparams['class']
    lr = hyperparams['lr']
    epochs = hyperparams['epochs']
    batch_size = hyperparams['batch_size']
    lambda_ = hyperparams['lambda']

    # this determines whether we find a constant-sum polymatrix (CSP) game or a zero-sum polymatrix game (ZSP) 
    global get_util
    if game_class == 'csp':
        get_util = utils.get_csp_util
        init_poly = utils.init_csp_poly
    elif game_class == 'zsp':
        get_util = utils.get_zsp_util
        init_poly = utils.init_zsp_poly
    else:
        raise AssertionError

    u_poly = init_poly(num_players, num_terminals) # initialize the game's utility function


    # dict for logging
    logging = {
        'gamma': [],
        'delta': [],
        'hyperparams': hyperparams,
        'losses': [],
        'bounds': []
    }

    # We use sequence-form LPs for computing best responses at the start of each epoch. 
    # The following line assembles LPs (in Gurobi) for this purpose. We get large efficiency gains by caching the solvers. 
    solvers = utils.assemble_solvers(game, num_players)

    # We parallelize and jit the functions for sampling, normalizing gradients and computing loss functions
    get_samples = jax.jit(utils.sample_strats, static_argnums=[1, 2, 3])
    get_loss =  jax.jit(jax.vmap(jax.value_and_grad(utils.loss), in_axes=(None, 
        None, 1, 1, 1, None, None), out_axes=0), static_argnums=6)   

    start = time.time() # for tracking run time
    step_size_schedule = [2**(-i) for i in range(6, 18)] # the step-size schedule. The step size decays every decay_rate epochs
    decay_rate = 5 # the number of epochs that must elapse before the step size decays
    ss_idx = 0 # index of the step size schefule

    # In addition to 
    bounds_params = []
    bound_list = []

    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            # At the start of an epoch, get the best responses.
            brs, gamma = utils.add_brs(u_poly, solvers, num_players, strats, get_util)
            L_sum = 0 # sum of losses for the iteration, used for logging
            lr = step_size_schedule[ss_idx] # the current learning rate
        
            # Next, we partition the dataset into batches, which are iterated over
            dataset_indices = [list(range(num_strats)) for i in range(num_players)]  # indices for each player
            dataset_indices = np.array(list((product(*dataset_indices)))) # cast to numpy
            # shuffle these indices
            shuffled_indices = jax.random.shuffle(key, dataset_indices, axis=0) # shuffle the indices
            _, key = jax.random.split(key)

            assert batch_size == num_strats # this is so parallelization in computing the loss works. In the future this constraint may be removed. This will require modifications to loss()
            num_matchups = shuffled_indices.shape[0] # the total number of match-ups
            # make sure all batches are of size batch_size
            if num_matchups%batch_size == 0:
                num_batches = num_matchups//batch_size
            else:
                num_batches = (num_matchups//batch_size)-1
                
            for batch_num in range(num_batches): # iterate over each batch
                start_idx = batch_num*batch_size # the starting index of the batch
                batch_indices = shuffled_indices[start_idx : start_idx+batch_size] #indices of the batch
                samples = get_samples(batch_indices, num_players, batch_size, num_terminals, strats) # samples is a sample of strategies from S^\times in the paper of size batch_size
           
            
                # Next, compute gradients and update u_poly
                L, grads = get_loss(u_poly, u_game, samples, brs, strats, lambda_, get_util) 
                grads = utils.tree_normalize(grads) 
                u_poly = jax.tree_map(
                    lambda p, g : p - lr *np.mean(g, axis=0), u_poly, grads)  
                
                # add the sum of losses 
                L_sum += np.sum(L)

            # logging 
            avg_loss =  L_sum/num_batches
            _ , gamma = utils.add_brs(u_poly, solvers, num_players, strats, get_util)
            delta = utils.get_delta(u_poly, u_game, strats, num_strats, get_util)
            bounds = 2*delta+(num_players-1)*gamma # bounds implied by theory
            bounds_params.append(u_poly)
            bound_list.append(bounds)
            pbar.set_postfix({'loss':avg_loss, 'gamma': gamma, 'delta':delta, 'bounds': bounds})
            logging['losses'].append(float(avg_loss))
            logging['bounds'].append(float(bounds))
            logging['gamma'].append(float(gamma))
            logging['delta'].append(float(delta))

            # decay step size
            if epoch % decay_rate == 0:
                ss_idx += 1
                ss_idx = min(ss_idx, len(step_size_schedule)-1)
                # take the model with best bounds when we move on to the next step size
                u_poly = bounds_params[np.argmin(np.array(bound_list))]
                bounds_params = []
                bound_list = []

    # final log
    logging['runtime'] = time.time() - start
    # compute gamma 
    _, gamma = utils.add_brs(u_poly, solvers, num_players, strats, get_util)
    logging['gamma'].append(float(gamma))
    # compute delta
    delta = utils.get_delta(u_poly, u_game, strats, num_strats, get_util)
    logging['delta'].append(float(delta))
    # compute the actual vulnerability
    vuln = utils.compute_vul(strats, u_game, num_players)
    logging['vuln'] = vuln

    json.dump(logging, open(save_dir/'log.json', 'w'))
import jax.numpy as np
import jax
from itertools import combinations

from multiprocessing import Pool
from sgdecompose.subgame_solver import SubgameSolver
from sgdecompose.sequence import construct_vars
from tqdm import tqdm


def sample_strats(rand_idxs, num_players, batch_size, num_terminals, strats):
    """
    Samples a random matchup of strategies. 

    args : 
        rand_idxs : the indices for each player
        num_players, batch_size, num_terminals : used for determining the size of the array
        strats : np array of cfr-generated strategies
    """
    samples = np.zeros((num_players, batch_size, num_terminals))
    samples = samples.at[0, :, :].set(strats[0, rand_idxs[:, 0], :])
    samples = samples.at[1, :, :].set(strats[1, rand_idxs[:, 1], :])
    samples = samples.at[2, :, :].set(strats[2, rand_idxs[:, 2], :])
    return samples


def tree_norm(x):
    return np.sqrt(jax.tree_util.tree_reduce(np.add, jax.tree_map(lambda x: np.sum(x**2), x)))


def tree_normalize(x):
    norm = tree_norm(x)
    if norm == 0: # avoid division by 0
        norm = 1
    
    return jax.tree_map(lambda x: x/norm, x)


def assemble_solvers(game, num_players):
    """
    Assembles the sequence-form LPs used for generating best responses
    """
    init = construct_vars(game, num_players, no_numpy=True) # this function returns necessary variables for sequence-form linear programs
    solvers = dict()
    for p_i in range(num_players):
        solvers[p_i] = SubgameSolver(game, p_i, init)
    return solvers
    

def subgames(num_players):
    return list(combinations(range(num_players), 2))


def loss(u_poly, u_game, sample, brs, strat_sample, lambda_, get_util):
    """
    Computes the loss for a batch. NOTE this will only work for 3 players. 

    args:
        u_poly : a dictionary of numpy arrays, representing the utility function of the polymatrix game
        u_game : a numpy array, representing the utility funciton of the original game
        sample : sample of strategies from the set of matchups
        brs : a numpy array of best responses
        strat_sample : sample of strategies from those CFR computed
        lambda_ : hyperparameter for weighting loss
        get_util : a function to get the utility funciton for subgames in the CSP game. 
    """
    # delta part of loss
    joint_reach_probs = (
        sample[0] * sample[1] * sample[2]
    )  # will only work for 3 palyers
    loss_delta = 0
    # hard-coded for now for 3 players
    u_01 = (sample[0] * sample[1]) @ get_util(0, 1, u_poly) 
    u_02 = (sample[0] * sample[2]) @ get_util(0, 2, u_poly) 
    u_0 = u_01 + u_02

    u_10 = (sample[1] * sample[0]) @ get_util(1, 0, u_poly) 
    u_12 = (sample[1] * sample[2]) @ get_util(1, 2, u_poly) 
    u_1 = u_10 + u_12

    u_20 = (sample[2] * sample[0]) @ get_util(2, 0, u_poly) 
    u_21 = (sample[2] * sample[1]) @ get_util(2, 1, u_poly) 
    u_2 = u_20 + u_21

    loss_delta += np.abs((joint_reach_probs @ u_game[0]) - u_0)
    loss_delta += np.abs((joint_reach_probs @ u_game[1]) - u_1)
    loss_delta += np.abs((joint_reach_probs @ u_game[2]) - u_2) 
    
    # gamma part of loss
    loss_gamma = 0
    u_01 = (strat_sample[0] * strat_sample[1]) @ get_util(0, 1, u_poly) 
    u_02 = (strat_sample[0] * strat_sample[2]) @ get_util(0, 2, u_poly) 

    u_01_dev = (brs[0] * strat_sample[1]) @ get_util(0, 1, u_poly) 
    u_02_dev = (brs[1] * strat_sample[2]) @ get_util(0, 2, u_poly) 

    loss_gamma += np.sum(jax.nn.relu(u_01_dev - u_01))
    loss_gamma += np.sum(jax.nn.relu(u_02_dev - u_02))

    # for player 1
    u_10 = (strat_sample[1] * strat_sample[0]) @ get_util(1, 0, u_poly) 
    u_12 = (strat_sample[1] * strat_sample[2]) @ get_util(1, 2, u_poly) 

    u_10_dev = (brs[2] * strat_sample[0]) @ get_util(1, 0, u_poly) 
    u_12_dev = (brs[3] * strat_sample[2]) @ get_util(1, 2, u_poly) 

    loss_gamma += np.sum(jax.nn.relu(u_10_dev - u_10))
    loss_gamma += np.sum(jax.nn.relu(u_12_dev - u_12))

    # for player 2
    u_20 = (strat_sample[2] * strat_sample[0]) @ get_util(2, 0, u_poly) 
    u_21 = (strat_sample[2] * strat_sample[1]) @ get_util(2, 1, u_poly) 

    u_20_dev = (brs[4] * strat_sample[0]) @ get_util(2, 0, u_poly) 
    u_21_dev = (brs[5] * strat_sample[1]) @ get_util(2, 1, u_poly) 

    loss_gamma += np.sum(jax.nn.relu(u_20_dev - u_20))
    loss_gamma += np.sum(jax.nn.relu(u_21_dev - u_21))

    loss_ =  lambda_*loss_delta + (1-lambda_)*loss_gamma # weighted sum of gamma and delta losses 
    return loss_


def get_delta(u_poly, u_game, strats, num_strats, get_util):
    """
    Returns the true value of delta. This funciton will only work for 3 players. 
    """
    get_delta_jit =  jax.jit(get_delta_, static_argnums=3)
    delta = 0
    for s_0 in range(0, num_strats):
        for s_1 in range(0, num_strats):
            samples = [
                strats[0, s_0, :],
                strats[1, s_1, :],
                strats[2, :, :]
            ]

            delta_ = get_delta_jit(u_poly, u_game, samples, get_util)
            delta = max(delta, np.max(delta_ ))
    return delta


def get_delta_(u_poly, u_game, sample, get_util):
    joint_reach_probs = (
        sample[0] * sample[1] * sample[2]
    )  
    # hard-coded for now for 3 players
    u_01 = (sample[0] * sample[1]) @ get_util(0, 1, u_poly) 
    u_02 = (sample[0] * sample[2]) @ get_util(0, 2, u_poly) 
    u_0 = u_01 + u_02 # utility of player 0 in the polymatrix game 

    u_10 = (sample[1] * sample[0]) @ get_util(1, 0, u_poly) 
    u_12 = (sample[1] * sample[2]) @ get_util(1, 2, u_poly) 
    u_1 = u_10 + u_12 # utility of player 1 in the polymatrix game 

    u_20 = (sample[2] * sample[0]) @ get_util(2, 0, u_poly) 
    u_21 = (sample[2] * sample[1]) @ get_util(2, 1, u_poly) 
    u_2 = u_20 + u_21 # utility of player 2 in the polymatrix game 

    delta_0 = np.abs((joint_reach_probs @ u_game[0]) - u_0)
    delta_1 = np.abs((joint_reach_probs @ u_game[1]) - u_1)
    delta_2 = np.abs((joint_reach_probs @ u_game[2]) - u_2) 

    deltas = np.array([delta_0, delta_1, delta_2])
    return np.max(deltas)

# We represent the utilty funciton as a dictionary, indexed by subgames. For each entry, there is a numpy array, which gives utilities at each terminal. 
# We give two implementations, one for constant-sum polymatrix (CSP) games and one for zero-sum polymatrix (ZSP) games. 


def init_zsp_poly(num_players, num_terminals):
    u_poly = dict()
    for (p_i, p_j) in subgames(num_players):
        u_poly[(p_i, p_j)] = np.zeros(num_terminals) # p_i's utility against p_j; p_j's will simply be the negative of p_i's. This logic is implemented in get_zsp_util()
    return u_poly


def init_csp_poly(num_players, num_terminals):
    u_poly = dict()
    for (p_i, p_j) in subgames(num_players):
        u_poly[(p_i, p_j)] = np.zeros(num_terminals)  # p_i's utility against p_j; p_j's will simply be the negative of p_i's. This logic is implemented in get_csp_util()
        u_poly[(p_i, p_j, 'c')] = 0.0 # the constant for the game between 
    return u_poly

# For convention, 

def get_zsp_util(p_i, p_j, u_poly):
    if p_i < p_j:
        u_ij = u_poly[(p_i, p_j)]
    else:
        u_ij = -u_poly[(p_j, p_i)]
    return u_ij


def get_csp_util(p_i, p_j, u_poly):
    if p_i < p_j:
        u_ij = u_poly[(p_i, p_j)]
    else:
        u_ij = u_poly[(p_j, p_i, 'c')] - u_poly[(p_j, p_i)]
    return u_ij


def add_brs(u_poly, solvers, num_players, strats, get_util): 
    """
    Generates the best-responses for each subgame at the start of each epoch.
    """
    gamma = 0 
    brs = []
    # First two loops iterate over subgames
    for p_i in range(num_players):
        for p_j in range(num_players):
            if p_i != p_j:
                brs_ij = []
                u_ij = get_util(p_i, p_j, u_poly)
                s_j_indices = strats.shape[1]
                for s_j_idx in range(s_j_indices):
                    s_i =  strats[p_i, s_j_idx, :] # strategy of p_i
                    s_j =  strats[p_j, s_j_idx, :] # strategy of p_j
                    br, gamma_ = solvers[p_i].subgame_br(u_ij, s_i, s_j)
                    brs_ij.append(br) 
                    gamma = max(gamma, gamma_)               
                brs.append(brs_ij)
   
    brs = np.array(brs)
    return brs, gamma


def compute_vul(strats, returns, num_players):
    """
    Computes the true vulnerability w.r.t. strats. Implements equation 10 (Vul_j) from the paper. 
    """
    vuln = 0
    for p_i in range(num_players):
        other_players = list(range(num_players))
        other_players.remove(p_i)
        p_j = other_players[0]
        p_k = other_players[1]
        for s_i_idx, s_i in enumerate(tqdm(strats[p_i])):
            intial_util =  (s_i
                    *strats[p_j][s_i_idx]
                    *strats[p_k][s_i_idx]) @ returns[p_i]
            reach1 = s_i  * strats[p_j] * returns[p_i]
            vuln = max(vuln, np.max(intial_util - reach1 @ strats[p_k].T))

    return float(vuln)
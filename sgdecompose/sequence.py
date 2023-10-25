# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Useful sequence form functions used in the MMD implementation."""

import numpy as np
from open_spiel.python import policy
from itertools import product

import gurobipy as gp
from gurobipy import GRB

def unique(list):
    unique_list = []
    for i in list:
        if i not in unique_list:
            unique_list.append(i)
    return unique_list


_EMPTY_INFOSET_KEYS = None
_EMPTY_INFOSET_ACTION_KEYS = None

_DELIMITER = " -=- "
def _get_isa_key(info_state, action):
    return info_state + _DELIMITER + str(action)


def _get_action_from_key(isa_key):
    _, action_str = isa_key.split(_DELIMITER)
    return int(action_str)


def _get_infostate_from_key(isa_key):
    assert not is_root(isa_key), "Cannot use this method for root nodes."
    infostate, _ = isa_key.split(_DELIMITER)
    return infostate


def is_root(key):
    return True if key in _EMPTY_INFOSET_KEYS + _EMPTY_INFOSET_ACTION_KEYS else False


def construct_vars(game, num_players, no_numpy=False):
    """Construct useful sequence from variables from game.

    Args:
        game: The spiel game. 
        num_players: the number of players in the game


    Returns:
        An 8 tuple of sequence form variables from _construct_vars by
        recursively
        traversing the game tree.

    """

    global _EMPTY_INFOSET_KEYS 
    global _EMPTY_INFOSET_ACTION_KEYS

    _EMPTY_INFOSET_KEYS = ["***EMPTY_INFOSET_P{}***".format(p) 
                           for p in range(num_players)]
    _EMPTY_INFOSET_ACTION_KEYS = [
        "***EMPTY_INFOSET_ACTION_P{}***".format(p) 
        for p in range(num_players)]
    
    initial_state = game.new_initial_state()

    # initialize variables
    infosets = [{_EMPTY_INFOSET_KEYS[p]: 0} for p in range(num_players)]
  
    infoset_actions_to_seq = [{_EMPTY_INFOSET_ACTION_KEYS[p]: 0} 
                              for p in range(num_players)]

    infoset_action_maps = [{_EMPTY_INFOSET_KEYS[p]: [_EMPTY_INFOSET_ACTION_KEYS[p]]} 
                           for p in range(num_players)]

    payoff_dict = dict()

    chance_dict= dict()

    infoset_parent_map = [{_EMPTY_INFOSET_KEYS[p]: _EMPTY_INFOSET_ACTION_KEYS[p]}
                            for p in range(num_players)]

    infoset_actions_children = [{_EMPTY_INFOSET_ACTION_KEYS[p]: []} 
                                  for p in range(num_players)]
    
    terminal_infoset_actions = [[] for i in range(num_players)] 
                                

    _construct_vars(initial_state, infosets, infoset_actions_to_seq,
                    infoset_action_maps, infoset_parent_map, 1.0,
                    _EMPTY_INFOSET_KEYS[:], _EMPTY_INFOSET_ACTION_KEYS[:],
                    payoff_dict, infoset_actions_children, num_players,  terminal_infoset_actions, chance_dict)

    if no_numpy == True:
        return (infosets, infoset_actions_to_seq,
                infoset_action_maps, infoset_parent_map,
                infoset_actions_children, payoff_dict, chance_dict)

    else:
        payoff_mat = _construct_numpy_vars(payoff_dict, infoset_actions_to_seq,  num_players)
        
        for p in range(num_players):
            terminal_infoset_actions[p] = unique(terminal_infoset_actions[p])
        
        return (infosets, infoset_actions_to_seq,
                infoset_action_maps, infoset_parent_map,
                payoff_mat, infoset_actions_children, payoff_dict, chance_dict)


def _construct_vars(state, infosets, infoset_actions_to_seq,
                    infoset_action_maps, infoset_parent_map, chance_reach,
                    parent_is_keys, parent_isa_keys, payoff_dict,
                    infoset_actions_children, num_players,  terminal_infoset_actions, chance_dict):
    """Recursively builds maps and the sequence form payoff matrix.

    Args:
        state: pyspiel (OpenSpiel) state
        infosets: a list of dicts, one per player, that maps infostate to an id.
          The dicts are filled by this function and should initially only
          contain root values.
        infoset_actions_to_seq: a list of dicts, one per player, that maps a
          string of (infostate, action) pair to an id. The dicts are filled by
          this function and should inirially only contain the root values.
        infoset_action_maps: a list of dicts, one per player, that maps each
          info_state to a list of (infostate, action) string.
        infoset_parent_map: a list of dicts, one per player, that maps each
          info_state to an (infostate, action) string.
        chance_reach: the contribution of chance's reach probability (should
          start at 1).
        parent_is_keys: a list of parent information state keys for this state
        parent_isa_keys: a list of parent (infostate, action) keys
        payoff_dict: a dict that maps ((infostate, action), (infostate, action))
          to the chance weighted reward
        infoset_actions_children: a list of dicts, one for each player, mapping
          (infostate, action) keys to reachable infostates for each player
        num_players: the number of players in the game
        terminal_infoset_actions: list of terminal sequences for each player
        chance_reach: chance's contribution to reaching each terminal. 

        
    """

    if state.is_terminal():
        returns = state.returns()
        matrix_index = [parent_isa_keys[p] for p in range(num_players)]
        matrix_index = tuple(matrix_index)
        payoff_dict.setdefault(matrix_index, np.zeros_like(returns))
        # note the payoff matrix A is for the min max problem x.T @ A y
        # where x is player 0 in openspiel
        for p in range(num_players):
            terminal_infoset_actions[p].append(parent_isa_keys[p])

        payoff_dict[matrix_index] += np.array(returns) #* chance_reach

        chance_dict[matrix_index] = chance_reach
        return

    if state.is_chance_node():
        for action, prob in state.chance_outcomes():
            new_state = state.child(action)
            _construct_vars(new_state, infosets, infoset_actions_to_seq,
                            infoset_action_maps, infoset_parent_map,
                            prob * chance_reach, parent_is_keys, parent_isa_keys,
                            payoff_dict, infoset_actions_children, num_players,
                            terminal_infoset_actions,  chance_dict)
        return

    player = state.current_player()
    info_state = state.information_state_string(player)
    legal_actions = state.legal_actions(player)

    # Add to the infostate maps
    if info_state not in infosets[player]:
        infosets[player][info_state] = len(infosets[player])
    if info_state not in infoset_action_maps[player]:
        infoset_action_maps[player][info_state] = []

    # Add to infoset to parent infoset action map
    if info_state not in infoset_parent_map[player]:
        infoset_parent_map[player][info_state] = parent_isa_keys[player]

    # add as child to parent
    if parent_isa_keys[player] in infoset_actions_children[player]:
        if info_state not in infoset_actions_children[player][
                parent_isa_keys[player]]:
            infoset_actions_children[player][parent_isa_keys[player]].append(
                info_state)
    else:
        infoset_actions_children[player][parent_isa_keys[player]] = [
            info_state]

    new_parent_is_keys = parent_is_keys[:]
    new_parent_is_keys[player] = info_state

    for action in legal_actions:
        isa_key = _get_isa_key(info_state, action)
        if isa_key not in infoset_actions_to_seq[player]:
            infoset_actions_to_seq[player][isa_key] = len(
                infoset_actions_to_seq[player])
        if isa_key not in infoset_action_maps[player][info_state]:
            infoset_action_maps[player][info_state].append(isa_key)

        new_parent_isa_keys = parent_isa_keys[:]
        new_parent_isa_keys[player] = isa_key
        new_state = state.child(action)
        _construct_vars(new_state, infosets, infoset_actions_to_seq,
                        infoset_action_maps, infoset_parent_map, chance_reach,
                        new_parent_is_keys, new_parent_isa_keys, payoff_dict,
                        infoset_actions_children, num_players, terminal_infoset_actions, chance_dict)


def _construct_numpy_vars(payoff_dict, infoset_actions_to_seq, num_players):
    """Convert sequence form payoff dict to numpy array.

    Args:
        payoff_dict: a dict that maps ((infostate, action), (infostate, action))
          to the chance weighted reward.
        infoset_actions_to_seq: a list of dicts, one per player, that maps a
          string of (infostate, action) pair to an id.

    Returns:
        A numpy array corresponding to the chance weighted rewards
        i.e. the sequence form payoff matrix.

    """
    sequence_sizes = tuple([len(infoset_actions_to_seq[p]) for p in range(num_players)] + [num_players])
    payoff_mat = np.zeros(sequence_sizes)
    sequences = [list(infoset_actions_to_seq[p].keys()) for p in range(num_players)]
    seq_prod = product(*sequences)

    for joint_seq in seq_prod:
        payoff_mat_index = []
        for p in range(num_players):
            payoff_mat_index.append(infoset_actions_to_seq[p][joint_seq[p]])
        payoff_mat_index = tuple(payoff_mat_index) 
        payoff_mat[payoff_mat_index, :] = payoff_dict.get(joint_seq, 0)
    
    return payoff_mat


def sequence_to_policy(sequences, game, infoset_actions_to_seq,
                       infoset_action_maps):
    """Convert sequence form policies to the realization-equivalent tabular ones.

    Args:
        sequences: list of two sequence form policies, one for each player.
        game: a spiel game with two players.
        infoset_actions_to_seq: a list of dicts, one per player, that maps a
          string of (infostate, action) pair to an id.
        infoset_action_maps: a list of dicts, one per player, that maps each
          info_state to a list of (infostate, action) string.

    Returns:
        A TabularPolicy object.
    """

    policies = policy.TabularPolicy(game)
    for player in range(2):
        for info_state in infoset_action_maps[player]:
            if is_root(info_state):
                continue

            state_policy = policies.policy_for_key(info_state)
            total_weight = 0
            num_actions = 0

            for isa_key in infoset_action_maps[player][info_state]:
                total_weight += sequences[player][infoset_actions_to_seq[player]
                                                  [isa_key]]
                num_actions += 1

            unif_pr = 1.0 / num_actions
            for isa_key in infoset_action_maps[player][info_state]:
                rel_weight = sequences[player][infoset_actions_to_seq[player][isa_key]]
                _, action_str = isa_key.split(_DELIMITER)
                action = int(action_str)
                pr_action = rel_weight / total_weight if total_weight > 0 else unif_pr
                state_policy[action] = pr_action
    return policies


def policy_to_sequence(game, policies, infoset_actions_to_seq, num_players):
    """Converts a TabularPolicy object for a two-player game.

    The converted policy is its realization-equivalent sequence form one.

    Args:
        game: a two-player open spiel game.
        policies: a TabularPolicy object.
        infoset_actions_to_seq: a list of dicts, one per player, that maps a
          string of (infostate, action) pair to an id.

    Returns:
        A list of numpy arrays, one for each player.
    """
    initial_state = game.new_initial_state()
    sequences = [
        np.ones(len(infoset_actions_to_seq[i])) for i in range(num_players)
    ]
    _policy_to_sequence(initial_state, policies, sequences,
                        infoset_actions_to_seq, [1 for i in range(num_players)])
    return sequences


def _policy_to_sequence(state, policies, sequences, infoset_actions_to_seq,
                        parent_seq_val):
    """Converts a TabularPolicy object to its equivalent sequence form.

    This method modifies the sequences inplace and should not be called directly.

    Args:
        state: an openspiel state.
        policies: a TabularPolicy object.
        sequences: list of numpy arrays to be modified.
        infoset_actions_to_seq: a list of dicts, one per player, that maps a
          string of (infostate, action) pair to an id.
        parent_seq_val: list of parent sequence values, this method should be
          called with initial value of [1*N].
    """

    if state.is_terminal():
        return

    if state.is_chance_node():
        for action, _ in state.chance_outcomes():
            new_state = state.child(action)
            _policy_to_sequence(new_state, policies, sequences,
                                infoset_actions_to_seq, parent_seq_val)
        return

    player = state.current_player()
    info_state = state.information_state_string(player)
    legal_actions = state.legal_actions(player)
    state_policy = policies.policy_for_key(info_state)
    for action in legal_actions:
        isa_key = _get_isa_key(info_state, action)
        # update sequence form
        sequences[player][infoset_actions_to_seq[player]
                          [isa_key]] = parent_seq_val[player] * state_policy[action]
        new_parent_seq_val = parent_seq_val[:]
        new_parent_seq_val[player] = sequences[player][
            infoset_actions_to_seq[player][isa_key]]
        new_state = state.child(action)
        _policy_to_sequence(new_state, policies, sequences, infoset_actions_to_seq,
                            new_parent_seq_val)
        

def player_policy_to_sequence(game, policy, player, infoset_actions_to_seq):
    """Converts a TabularPolicy object for a two-player game.

    The converted policy is its realization-equivalent sequence form one.

    NOTE: this version is meant for a single player.

    Args:
        game: an open spiel game.
        policies: a TabularPolicy object.
        infoset_actions_to_seq: a list of dicts, one per player, that maps a
          string of (infostate, action) pair to an id.

    Returns:
        A list of numpy arrays, one for each player.
    """
    initial_state = game.new_initial_state()
  
    sequence = np.ones(len(infoset_actions_to_seq[player])) 

    _player_policy_to_sequence(initial_state, policy, player, sequence, infoset_actions_to_seq, 1)
    return sequence


def _player_policy_to_sequence(state, policy, seq_player,  sequence, infoset_actions_to_seq, parent_seq_val):
    """Converts a TabularPolicy object to its equivalent sequence form.

    This method modifies the sequences inplace and should not be called directly.

    Args:
        state: an openspiel state.
        policy: a TabularPolicy object.
        seq_player: integer of the player
        sequences: list of numpy arrays to be modified.
        infoset_actions_to_seq: a list of dicts, one per player, that maps a
          string of (infostate, action) pair to an id.
        parent_seq_val: list of parent sequence values, this method should be
          called with initial value of [1*N].
    """

    if state.is_terminal():
        return

    if state.is_chance_node():
        for action, _ in state.chance_outcomes():
            new_state = state.child(action)
            _player_policy_to_sequence(new_state, 
                                       policy, 
                                       seq_player, 
                                       sequence,
                                       infoset_actions_to_seq, 
                                       parent_seq_val)
        return

    player = state.current_player()
    info_state = state.information_state_string(player)
    legal_actions = state.legal_actions(player)
    
    
    if player == seq_player:
        state_policy = policy.policy_for_key(info_state)
        for action in legal_actions:
            isa_key = _get_isa_key(info_state, action)
            # update sequence form
            sequence[infoset_actions_to_seq[player][isa_key]] = parent_seq_val * state_policy[action]
            # new_parent_seq_val = parent_seq_val
            new_parent_seq_val = sequence[infoset_actions_to_seq[player][isa_key]]
            new_state = state.child(action)
            _player_policy_to_sequence(new_state,
                                policy, 
                                seq_player, 
                                sequence, 
                                infoset_actions_to_seq,
                                new_parent_seq_val)
    else:
        for action in legal_actions:
            new_state = state.child(action)
            _player_policy_to_sequence(new_state,
                                policy, 
                                seq_player, 
                                sequence, 
                                infoset_actions_to_seq,
                                parent_seq_val)


def compute_br(real_plan, p_i, terminal_seqs, terminal_returns, infoset_actions_to_seq, chance_dict, infosets, infoset_action_maps, infoset_parent_map, num_players):
    m = gp.Model("BR")
    m.Params.LogToConsole = 0
    real_plan_i =  m.addMVar(real_plan[p_i].shape, name='real_plan') # the realization plan of player p_i, the decision variables

    # reach_probs is the reach probabiliy of each terminal sequence 
    reach_probs_minus_i = np.ones(len(terminal_seqs)) 
    for i, seq in enumerate(terminal_seqs):
        for p_j in range(num_players):
            if p_j != p_i:
                reach_probs_minus_i[i] *= real_plan[p_j][infoset_actions_to_seq[p_j][seq[p_j]]]
        reach_probs_minus_i[i] *= chance_dict[seq]

    idxs = [infoset_actions_to_seq[p_i][seq[p_i]] for seq in terminal_seqs]  # for each terminal sequence, which one of p_i's sequences was played
    objective = (terminal_returns*reach_probs_minus_i)@real_plan_i[idxs]
    m.setObjective(objective, GRB.MAXIMIZE)

    # constraints
    empty_seq = "***EMPTY_INFOSET_ACTION_P{}***".format(p_i) 
    empty_seq_idx = infoset_actions_to_seq[p_i][empty_seq]
    m.addConstr(real_plan_i[empty_seq_idx] == 1)

    for infoset in infosets[p_i]:
        extend_seqs = infoset_action_maps[p_i][infoset] #the set of sequences that extend infoset
        extend_seq_idxs = [infoset_actions_to_seq[p_i][seq] for seq in extend_seqs]
        parent_seq = infoset_parent_map[p_i][infoset] # the unique sequence leading to infoset
        parent_seq_idx = infoset_actions_to_seq[p_i][parent_seq] 
        m.addConstr(real_plan_i[parent_seq_idx] == real_plan_i[extend_seq_idxs].sum())
        
    m.optimize()
    obj = m.getObjective()
    return obj.getValue()
    

def chance_weighted_br(real_plan, p_i, terminal_seqs, terminal_returns, infoset_actions_to_seq, infosets, infoset_action_maps, infoset_parent_map, num_players):
    # uses sequence-form to compute a best-response with linear programming

    m = gp.Model("BR")
    m.Params.LogToConsole = 0
    real_plan_i =  m.addMVar(real_plan[p_i].shape, name='real_plan') # the realization plan of player p_i, the decision variables

    # reach_probs is the reach probabiliy of each terminal sequence 
    reach_probs_minus_i = np.ones(len(terminal_seqs)) 
    for i, seq in enumerate(terminal_seqs):
        for p_j in range(num_players):
            if p_j != p_i:
                reach_probs_minus_i[i] *= real_plan[p_j][infoset_actions_to_seq[p_j][seq[p_j]]]
        # reach_probs_minus_i[i] *= chance_dict[seq]

    idxs = [infoset_actions_to_seq[p_i][seq[p_i]] for seq in terminal_seqs]  # for each terminal sequence, which one of p_i's sequences was played
    objective = (terminal_returns*reach_probs_minus_i)@real_plan_i[idxs]
    m.setObjective(objective, GRB.MAXIMIZE)

    # constraints
    empty_seq = "***EMPTY_INFOSET_ACTION_P{}***".format(p_i) 
    empty_seq_idx = infoset_actions_to_seq[p_i][empty_seq]
    m.addConstr(real_plan_i[empty_seq_idx] == 1)

    for infoset in infosets[p_i]:
        extend_seqs = infoset_action_maps[p_i][infoset] #the set of sequences that extend infoset
        extend_seq_idxs = [infoset_actions_to_seq[p_i][seq] for seq in extend_seqs]
        parent_seq = infoset_parent_map[p_i][infoset] # the unique sequence leading to infoset
        parent_seq_idx = infoset_actions_to_seq[p_i][parent_seq] 
        m.addConstr(real_plan_i[parent_seq_idx] == real_plan_i[extend_seq_idxs].sum())
        
    m.optimize()
    obj = m.getObjective()
    return obj.getValue()
    # uses sequence-form to compute a best-response with linear programming
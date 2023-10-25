import jax.numpy as np

def get_terminal_returns(game):
    """
    Retuns an np.array of all the retuns of terminal states

    args:
        game : openspiel game

    returns:
       returns : (np.Array) returns at each terminals of the game
    """

    terminals = []

    def recurse(state):
        if state.is_terminal():
            nonlocal terminals
            terminals.append(state.returns())
            return

        if state.is_chance_node():
            for action, action_prob in state.chance_outcomes():
                new_state = state.child(action)
                recurse(new_state)
            return

        for action in state.legal_actions():
            new_state = state.child(action)
            recurse(new_state)

    recurse(game.new_initial_state())
    return np.array(terminals).T


def player_terminal_reach(game, policy, player):
    """
    Returns a vector containing the reach probabilities of each terminal history as if player
    were following policy and other players are playing to that history

    args:
        game : openspiel game
        policy : (openspiel.TabularPolicy) a (potentially joiny) policy (behavior strategy) profile that includes at least 
            all of player's infosets
        player : (int) the player in question

    returns:
        reach_probs : (np.array) reach probability of each terminal history 
    """

    reach_probs = list()

    def recurse(state, reach_prob):
        if state.is_terminal():
            nonlocal reach_probs
            reach_probs.append(reach_prob)
            return

        if state.is_chance_node():
            for action, action_prob in state.chance_outcomes():
                new_state = state.child(action)
                recurse(new_state, reach_prob)
            return


        action_probabilities = policy.action_probabilities(state)
        for action in state.legal_actions():
            new_state = state.child(action)
            new_reach_prob = reach_prob 
            
            nonlocal player
            current_player = state.current_player()
            if current_player == player:
                action_prob = action_probabilities[action]
                new_reach_prob *= action_prob
               
            recurse(new_state, new_reach_prob)

    recurse(game.new_initial_state(), 1)
    return np.array(reach_probs)


def chance_terminal_reach_probs(game):
    """
    Returns a vector containing the reach probabilities of each terminal history as if chance
    were following policy and other players are playing to that history

    args:
        game : openspiel game

    returns:
        reach_probs : (np.array) reach probability of each terminal history 
    """

    reach_probs = list()

    def recurse(state, reach_prob):
        if state.is_terminal():
            nonlocal reach_probs
            reach_probs.append(reach_prob)
            return

        if state.is_chance_node():
            for action, action_prob in state.chance_outcomes():
                new_state = state.child(action)
                recurse(new_state, reach_prob*action_prob)
            return

        for action in state.legal_actions():
            new_state = state.child(action)
            recurse(new_state, reach_prob)
    
    recurse(game.new_initial_state(), 1)
    return np.array(reach_probs)


def evaluate_br(state, policy, br_policy, num_players, br_player):
    '''
    Evaluates utility of a policy.
    '''

    if state.is_terminal():
        return np.asarray(state.returns())

    if state.is_chance_node():
        state_value = 0.0
        for action, action_prob in state.chance_outcomes():
            new_state = state.child(action)
            state_value += action_prob * evaluate_br( new_state , policy, br_policy, num_players, br_player)
        return state_value
    
    state_value = np.zeros(num_players)

    current_player = state.current_player()
    if current_player==br_player:
        info_state_policy =br_policy(state)
    else:

        info_state_policy = policy(state)
    for action in state.legal_actions():
        action_prob = info_state_policy.get(action, 0.)

        if action_prob != 0:
            new_state = state.child(action)
            child_utility = evaluate_br(new_state, policy, br_policy, num_players, br_player)
            state_value += action_prob * child_utility

    return state_value

import gurobipy as gp
from gurobipy import GRB
import jax.numpy as np
import numpy as onp


class SubgameSolver:
    """
    A helper class for computing BRs in subgames as the first step in each epoch of SGDecompose.

    NOTE: __init__() constructs an LP without an objective, these constaints are cached. The objective is added in subgame_br(). Caching the constraints makes things a lot more efficient. 
    """

    def __init__(self, game, p_i, init):
        self.game = game
        (self.infosets, self.infoset_actions_to_seq,
         self.infoset_action_maps, self.infoset_parent_map,
         self.infoset_actions_children, self.payoff_dict, self.chance_dict) = init
    
        self.terminal_seqs = list(self.payoff_dict.keys())
        self.p_i = p_i

        self.idxs = [self.infoset_actions_to_seq[self.p_i][seq[self.p_i]]
                for seq in self.terminal_seqs]

        # initialize the model
        self.m = gp.Model("BR")
        self.m.Params.LogToConsole = 0
        self.real_plan_i = self.m.addMVar(
                self.get_sequence_shape(self.p_i).shape, name='real_plan') # player i's realization plan

        # constraints
        empty_seq = "***EMPTY_INFOSET_ACTION_P{}***".format(self.p_i)
        empty_seq_idx = self.infoset_actions_to_seq[self.p_i][empty_seq]
        self.m.addConstr(self.real_plan_i[empty_seq_idx] == 1)

        for infoset in self.infosets[self.p_i]:
            # the set of sequences that extend infoset
            extend_seqs = self.infoset_action_maps[self.p_i][infoset]
            extend_seq_idxs = [self.infoset_actions_to_seq[self.p_i][seq]
                            for seq in extend_seqs]
            # the unique sequence leading to infoset
            parent_seq = self.infoset_parent_map[self.p_i][infoset]
            parent_seq_idx = self.infoset_actions_to_seq[self.p_i][parent_seq]
            self.m.addConstr(self.real_plan_i[parent_seq_idx] ==
                        self.real_plan_i[extend_seq_idxs].sum())
            

    def get_sequence_shape(self, p_i):
        return np.ones(len(self.infoset_actions_to_seq[p_i]))


    def subgame_br(self, u_ij, s_i, s_j):
        '''
        Computes a best response (br) for p_i in the subgame against p_j
        '''
        # cast to numpy for use with gurobi
        u_ij = onp.asarray(u_ij)
        s_j = onp.asarray(s_j)

        objective = u_ij @ (s_j * self.real_plan_i[self.idxs])

        self.m.setObjective(objective, GRB.MAXIMIZE)
        self.m.optimize()       
        reach_probs_i = self.real_plan_i.X[self.idxs]

        u_2 = (reach_probs_i * s_j) @ u_ij # utility after deviating
        u_1 = (s_i * s_j) @ u_ij # utility prior to deviationg
        incentive = u_2 - u_1
        
        return reach_probs_i, incentive

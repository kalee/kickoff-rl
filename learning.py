# -*- coding: utf-8 -*-

import numpy as np


class QLearningGW :
    def __init__(self,env, alpha=0.1, gamma=0.99, epsilon=0.1) :
        # assume env is a GridWorld object
        
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.num_actions = 3
        self.num_states = env.width
        self.q_table = np.zeros((self.num_states, self.num_actions))
        
        
    def greedyAction(self,Q,s) :
        '''
        Q is a Q table; 
        s is a state index;
        this method returns the column index corresponding to the largest element on row s of Q
        '''
        
        return 0
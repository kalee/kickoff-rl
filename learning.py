# -*- coding: utf-8 -*-

import numpy as np


class QLearningGW :
    '''
    This is hard-coded for the GridWorld environment and will not work with the inertial version.
    '''
    
    
    def __init__(self,env, alpha=0.5, gamma=0.99, epsilon=0.1) :
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
    
    def randomAction(self) :
        return int(np.random.randint(0,3))
    
    def select_action(self,s) :
        '''
        return an action index given state s. 
        If epsilon-greedy, return a random action w.p. epsilon, otherwise a greedy action.
        '''
        # fill in code here!
    
    def update_q(self,s,a) :
        '''
        perform an update of self.q_table based on the transition starting at
        state s and applying action a.
        returns new state index s and done bit.
        '''
        
        self.env.state = s
        sprime,reward,done = self.env.step(a)
        # fill in code here!
        
        return sprime, done
    
    def learn(self, episodes=5000, max_steps=100):
        '''
        initialize the environment up to episodes times, run each training episode up to max_steps timesteps.
        Each timestep, get an action using the self.select_action method and perform an update on the q-table.
        If the self.update_q method returns done=True, start a new training episode.
        '''
        # fill in code here!
    
    
if __name__ == "__main__" :
    from environments import GridWorld
    
    gw = GridWorld(20)
    
    agent = QLearningGW(gw,alpha=.5)
    
    agent.learn()
    
    print(agent.q_table)

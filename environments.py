# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:15:50 2025

@author: phili
"""


import numpy as np
import matplotlib.pyplot as plt



class GridWorld :
    
    LEFT = 0
    COAST = 1
    RIGHT = 2
    
    def __init__(self,width) :
        self.width = width
        self.reset()
        
    def reset(self) :
        '''
        This method sets self.state to the initial state and then return the current state
        '''
        
        # initialize
        self.state = self.width - 1
        
        return self.state
    
    def step(self,action) :
        '''
        Input: action index
        Output: 
            - state after action
            - this stage reward
            - boolean: True if terminated, False if not terminated
        '''
        
        reward = 0
        # action penalties:
        if action == self.LEFT :
            self.state -= 1
            reward -= 1
        elif action == self.RIGHT :
            self.state += 1
            reward -= 1
        elif action == self.COAST :
            pass
        
        if self.state <= 0 :
            reward += 10
        
        # check termination:
        done = False
        if self.state <= 0 :
            done = True
        
        # don't run off the right side of the world:
        if self.state >= self.width :
            self.state = self.width-1
            
        return self.state, reward, done
    
    
if __name__ == '__main__' :
    
    gw = GridWorld(5)
    print(gw.reset())
    
    sample_actions = [0,0,1,2,0,0,0]
    for action in sample_actions :
        print(gw.step(action))
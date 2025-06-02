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
        
        self.reset()
        
    def reset(self) :
        '''
        This method sets self.state to the initial state and then return the current state
        '''
        self.state = None ## edit this line of code!
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
        done = False
            
        return self.state, reward, done
    
    
    



class GridWorldInertial :
    
    # interpret these as "thrust" events:
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
        
        self.state = {}
        
        # agent starts motionless at the far right:
        self.state['position'] = self.width - 1
        self.state['velocity'] = 0
        
        return self.state
    
    def step(self,action) :
        '''
        Input: action index
        Output: 
            - state after action
            - this stage reward
            - boolean: True if terminated, False if not terminated
        '''
        
        # interpretation: previous velocity propagates first, then is updated.
        self.state['position'] += self.state['velocity']
        
        
        # don't run off the right side of the world:
        if self.state['position'] >= self.width :
            self.state['position'] = self.width-1
            self.state['velocity'] = 0
        
        deltaV = action - 1 # shift to make this an actual signed number
        self.state['velocity'] += deltaV
        if self.state['velocity'] > 1 :
            self.state['velocity'] = 1
        if self.state['velocity'] < -1 :
            self.state['velocity'] = -1
        
        reward = 0
        # thrust penalties:
        if action == self.LEFT :
            reward -= 1
        elif action == self.RIGHT :
            reward -= 1
        elif action == self.COAST :
            pass
        
        if self.state['position'] <= 0 :
            reward += 10
        
        # check termination:
        done = False
        if self.state['position'] <= 0 :
            done = True
            
        return self.state, reward, done
    
    
if __name__ == '__main__' :
    
    gw = GridWorld(5)
    print(gw.reset())
    
    sample_actions = [0,0,1,2,0,0,0]
    for action in sample_actions :
        print(gw.step(action))
    
    
if __name__ == '__main__' :
    
    gwi = GridWorldInertial(5)
    print(gwi.reset())
    
    sample_actions_gwi = [0,0,2,2,1,1,0,1,1,1,1]
    for action in sample_actions_gwi :
        print(gwi.step(action))
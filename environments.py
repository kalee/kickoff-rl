# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:15:50 2025
Markov Decision Processes (MDPs) 
@author: phili
"""

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



class GridWorld(gym.Env) :
    
    metadata = {"render_modes": ['rgb_array']}
    
    LEFT = 0
    COAST = 1
    RIGHT = 2
    
    def __init__(self, width=5, render_mode="rgb_array"):
        self.render_mode = render_mode
        self.width = width
        self.action_space = spaces.Discrete(3)  # the three actions
        
        self.observation_space = spaces.Discrete(self.width)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        '''
        This method sets self.state to the initial state and then return the current state
        '''
        
        self.state = self.width - 1  ## edit this line of code!
        
        return self.state
    
    def step(self, action):
        '''
        Input: action index
        Output: 
            - state after action
            - this stage reward
            - boolean: True if terminated, False if not terminated
        '''
        # Currently the actions are 0 - left, 1 - coast, and 2 - right.
        # It is easier to sum these to visualize left or right if those values
        # are -1 and 1.  This creates the value motion to allow for that.
        motion = action - 1
        
        # Setup so that the current State or Position in 1-D representation
        # is bounded by the number of cells, and is zero indexed. this keeps 
        # the state bounded by 0 and width - 1.  Width -1 to map counting to 
        # zero based index to reference location.
        newState = max(0, min(self.width - 1, self.state + motion))
        
        # Determine Cost function defined by reward for making a move, or for
        # moving into the target position.
        
        # Default for not moving
        reward = 0
        
        # Reward when moving into cell indexed by zero
        if newState == 0:
            reward += 10
        
        # Reward/Cost for left or right action
        if action == self.LEFT or action == self.RIGHT:
            reward -= 1
        
        # Default value set
        done = False
        
        # Update for terminal condition
        if newState == 0:
            done = True
        
        # Update the state    
        self.state = newState
        
        # returns state, reward, done, boolean, dictionary
        # The boolean and dictionary are used when step functionality
        # is extended during learning calls.
        return self.state, reward, done, False, {}
    
    
    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array', "Only 'rgb_array' mode is supported"
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(self.width, 1))
        canvas = FigureCanvas(fig)
    
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
        position = self.state
    
        # Draw background grid
        for i in range(self.width):
            rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, edgecolor='gray', facecolor='white')
            ax.add_patch(rect)
    
        # Draw target at far left
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, color='red', alpha=0.3))
    
        # Draw agent
        ax.add_patch(patches.Rectangle((position, 0), 1, 1, color='blue'))
    
    
        
        # Render to RGB array
        canvas.draw()
        buf = canvas.buffer_rgba()  # use RGBA buffer
        width, height = fig.get_size_inches() * fig.dpi
        image = np.frombuffer(buf, dtype=np.uint8).reshape(int(height), int(width), 4)[..., :3]  # strip alpha
        
        plt.close(fig)
    
        return image
    
    
    
if __name__ == '__main__' :
    
    # Test - Create a 1D set of five items, indexed from 0 to 4
    gw = GridWorld(5)

    print(gw.reset())
    

    sample_actions = [0,0,1,2,0,0,0,1]
    cumulative_reward = 0
    for action in sample_actions:
        state, rew, done, _, _ = gw.step(action)
        print(state, rew, done)
        cumulative_reward += rew
        if done:
            break
    
    print(cumulative_reward)
       
    

"""    
class GridWorldInertial :
    
    # interpret these as "thrust" events:
    LEFT = 0
    COAST = 1
    RIGHT = 2
    
    def __init__(self, width) :
        self.reset()
        
    def reset(self) :
        '''
        This method sets self.state to the initial state and then return the current state
        '''
        
        # initialize
        self.state = {}
        
        # agent starts motionless at the far right:
        self.state['position'] = None
        self.state['velocity'] = None
        
        return self.state
    
    def step(self, action) :
        '''
        Input: action index
        Output: 
            - state after action
            - this stage reward
            - boolean: True if terminated, False if not terminated
        '''
        
        # interpretation: previous velocity propagates first, then is updated.
        
        
        # don't run off the right side of the world!
        
        # calculate rewards:
        reward = 0
        
        # check termination:
        done = False
            
        return self.state, reward, done
    
    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array', "Only 'rgb_array' mode is supported"
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(self.width, 1))
        canvas = FigureCanvas(fig)
    
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
        position = self.state['position']
        velocity = self.state['velocity']
    
        # Draw background grid
        for i in range(self.width):
            rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, edgecolor='gray', facecolor='white')
            ax.add_patch(rect)
    
        # Draw target at far left
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, color='red', alpha=0.3))
    
        # Draw agent
        ax.add_patch(patches.Rectangle((position, 0), 1, 1, color='blue'))
    
        # Draw velocity arrow (rightward or leftward)
        if velocity != 0:
            direction = np.sign(velocity)
            arrow_length = min(abs(velocity), self.width - 1)
            arrow_x = position + 0.5
            arrow_dx = direction * arrow_length
            ax.arrow(
                arrow_x, 0.5,  # Start at center of agent cell
                dx=arrow_dx, dy=0,
                width=0.05, head_width=0.3, head_length=0.3,
                length_includes_head=True, color='black'
            )
    
        
        # Render to RGB array
        canvas.draw()
        buf = canvas.buffer_rgba()  # use RGBA buffer
        width, height = fig.get_size_inches() * fig.dpi
        image = np.frombuffer(buf, dtype=np.uint8).reshape(int(height), int(width), 4)[..., :3]  # strip alpha
        
        plt.close(fig)
    
        return image



if __name__ == '__main__' :
    
    gwi = GridWorldInertial(5)
    print(gwi.reset())
    
    sample_actions_gwi = [0,0,2,2,1,1,0,1,1,1,1]
    for action in sample_actions_gwi :
        print(gwi.step(action))
"""


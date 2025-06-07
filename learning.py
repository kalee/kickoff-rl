# -*- coding: utf-8 -*-
"""
LLM - [Perplexity]
In the Q-table, you'll observe the estimated optimal Q-values for each 
state-action pair. Over time, the Q-table converges to reflect the expected 
long-term reward for each action in each state, given the agent's policy. 
The agent learns to choose actions that maximize its cumulative reward, 
balancing exploration and exploitation.
"""

"""
LLM - [Perplexity]
Gamma (γ = 0.1): With a low discount factor, the agent focuses on immediate 
    rewards rather than future ones. This means the agent will be more 
    reactive to the immediate consequences of its actions.
Epsilon (ε = 0.1): This parameter controls exploration. With a small epsilon, 
    the agent primarily exploits its current knowledge (i.e., chooses actions 
    with the highest Q-values) but occasionally explores random actions. This 
    helps the agent discover potentially better strategies.
Alpha (α = 0.1): The learning rate determines how quickly the Q-values are 
    updated. A smaller alpha means the agent updates its Q-values more 
    conservatively, which can lead to more stable learning but may also 
    slow down the learning process.
"""

import numpy as np


class QLearningGW :
    '''
    This is hard-coded for the GridWorld environment and will not work with the inertial version.
    '''
    
    
    def __init__(self, env, alpha=0.5, gamma=0.99, epsilon=0.1) :
        # assume env is a GridWorld object
        
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.num_actions = 3
        self.num_states = env.width
        self.q_table = np.zeros((self.num_states, self.num_actions))
        
        
    def greedyAction(self, Q, s) :
        '''
        Q is a Q table; 
        s is a state index;
        this method returns the column index corresponding to the largest element on row s of Q
        Q[s, :] represents the row at index s
        '''
 
        return np.argmax(Q[s, :])
        
        
    def randomAction(self) :
        return int(np.random.randint(0,3))

    
    def select_action(self, s) :
        '''
        return an action index given state s. 
        If epsilon-greedy, return a random action w.p. epsilon, otherwise a greedy action.
        '''
        # fill in code here!
        if np.random.rand() < self.epsilon:
            return self.randomAction()
        else:
            return self.greedyAction(self.q_table, s)
    
    
    def update_q(self, s, a) :
        '''
        perform an update of self.q_table based on the transition starting at
        state s and applying action a.
        returns new state index s and done bit.
        '''
        # After running step, sprime = new state
        self.env.state = s
        
        sprime, reward, done, _, _ = self.env.step(a)
        
        
        # fill in code here!
        oldQ = self.q_table[s, a]
        newQ = reward + self.gamma * np.max(self.q_table[sprime, :])
        
        self.q_table[s, a] = (1 - self.alpha) * oldQ + self.alpha * newQ
        
        return sprime, done
    
    
    def learn(self, episodes=500, max_steps=1000):
        '''
        initialize the environment up to episodes times, run each training episode up to max_steps timesteps.
        Each timestep, get an action using the self.select_action method and perform an update on the q-table.
        If the self.update_q method returns done=True, start a new training episode.
        '''
        # fill in code here!
        
        for _ in range(episodes):
            s = self.env.reset()
            # s = int(np.random.randint(1,self.env.width))
            # self.env.state = s
            done = False
            for _ in range(max_steps):
                a = self.select_action(s)
                sprime, done = self.update_q(s, a)
                s = sprime
                if done:
                    break


        """
        sprime, reward, done, _, _ = self.env.step(self.env.LEFT) 
        print(self.env.state)
        self.q_table[s, self.env.LEFT] = reward
        
        sprime, reward, done, _, _ = self.env.step(self.env.COAST)
        self.q_table[s, self.env.COAST] = reward
        print(self.env.state)
        
        # Can't just run this one, changes state/row.
        sprime, reward, done, _, _ = self.env.step(self.env.RIGHT)
        self.q_table[s, self.env.RIGHT] = reward
        print(self.env.state)
        """

# q_table = np.zeros((num_states, num_actions))    
if __name__ == "__main__" :
    from environments import GridWorld
    
    gw = GridWorld(5)
    # Alpha [0,1] Higher value, accept new value
    # Gamma [0,1] percentage of reward received 
    # Epsilon [0,1] Percent random vs greedy
    

    # Print the current state in gw
    print("state=",gw.state)
    
    agent = QLearningGW(gw, alpha=1, gamma=1, epsilon=0.1)
    
    #agent.learn(episodes=5000)
    agent.learn(episodes=500, max_steps=500)

    for i in range(3):
        agent.q_table[agent.env.state,i] = agent.q_table[agent.env.state+1,i] + 1


    
    print(agent.q_table)

"""
alpha=1
gamma=1
epsilon=0.8
Reloaded modules: environments
[[0. 0. 0.]
 [9. 9. 7.]
 [8. 8. 6.]
 [7. 7. 5.]
 [6. 6. 5.]]

alpha=1
gamma=.5
epsilon=1
%runfile /home/klee/srccode/uccs.edu/kickoff-rl/learning.py --wdir
Reloaded modules: environments
[[ 0.     0.     0.   ]
 [ 9.     4.5    0.75 ]
 [ 3.5    1.75  -0.625]
 [ 0.75   0.375 -1.   ]
 [-0.625  0.    -1.   ]]

alpha=1
gamma=1
epsilon=0.1
Reloaded modules: environments
[[0. 0. 0.]
 [9. 9. 7.]
 [8. 8. 6.]
 [7. 0. 5.]
 [6. 6. 5.]]
What needs to be updated to get the first line to be
9 10 8 ?

"""

"""

In [63]: %runfile /Users/klee/srccode/uccs.edu/kickoff-rl/learning.py --wdir
Reloaded modules: environments
state= 4
[[0. 0. 0.]
 [9. 9. 7.]
 [8. 0. 6.]
 [7. 7. 5.]
 [6. 6. 5.]]

In [64]: agent.update_q(0,0)
Out[64]: (0, True)

In [65]: print(agent.q_table)
[[9. 0. 0.]
 [9. 9. 7.]
 [8. 0. 6.]
 [7. 7. 5.]
 [6. 6. 5.]]

In [66]: agent.update_q(0,1)
Out[66]: (0, True)

In [67]: print(agent.q_table)
[[ 9. 19.  0.]
 [ 9.  9.  7.]
 [ 8.  0.  6.]
 [ 7.  7.  5.]
 [ 6.  6.  5.]]

In [68]: 

Why doesn't agent.update_q(0,1) put 10 in the q_table?

"""


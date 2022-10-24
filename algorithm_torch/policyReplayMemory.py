import random, os
import numpy as np

class policyReplayMemory:
    """Class for Replay Memory of Policy Network
    """
    def __init__(self, memory_size:int, batch_size:int):
        """[Initialisation Arguments]

        Args:
            memory_size ([int]): [Memory size to be allocated]
            batch_size ([int]): [Batch Size]
        """
        
        self.states = []
        self.neighbor_mask = []
        self.actions = []
        self.rewards = []
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    # Put data in policy replay memory
    def add(self, state:np.array, action:np.array, reward:list, mask:np.array):
        """Adds Experience to the object

        Args:
            state ([Numpy Array]): [State]
            action ([Numpy Array]): [Action]
            reward ([list]): [reward]
            mask ([Numpy Array]): [Mask]
        """
        
        if self.curr_lens == 0:
            self.states = state
            self.actions = action
            self.rewards = reward
            self.neighbor_mask = mask
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, state), axis=0)
            self.neighbor_mask = np.concatenate((self.neighbor_mask, mask), axis=0)
            self.actions = np.concatenate((self.actions, action), axis=0)
            self.rewards = np.concatenate((self.rewards, reward), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = state.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)
            self.states[index:(index + new_sample_lens)] = state
            self.actions[index:(index + new_sample_lens)] = action
            self.rewards[index:(index + new_sample_lens)] = reward
            self.neighbor_mask[index:(index + new_sample_lens)] = mask

    # Take a batch of samples
    def sample(self)->list:
        """Sample a batch of experience

        Returns:
            [list]: [Batch of experience]
        """
        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, np.array(self.rewards), self.neighbor_mask]
        indices = random.sample(list(range(0, self.curr_lens)), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_mask = self.neighbor_mask[indices]
        return [batch_s, batch_a, batch_r, batch_mask]

    def reset(self)->None:
        """reset the variables
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.neighbor_mask = []
        self.curr_lens = 0 
        
        

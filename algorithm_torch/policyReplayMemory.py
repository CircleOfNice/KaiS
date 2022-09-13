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
    def add(self, s:np.array, a:np.array, r:list, mask:np.array):
        """Adds Experience to the object

        Args:
            s ([Numpy Array]): [State]
            a ([Numpy Array]): [Action]
            r ([list]): [reward]
            mask ([Numpy Array]): [Mask]
        """
        
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.neighbor_mask = mask
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s), axis=0)
            self.neighbor_mask = np.concatenate((self.neighbor_mask, mask), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)
            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
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
        
        

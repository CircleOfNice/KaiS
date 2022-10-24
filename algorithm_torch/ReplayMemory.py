import random, os
import numpy as np
class ReplayMemory:
    """Class for replay memory
    """
    def __init__(self, memory_size:int, batch_size:int):
        """Replay Memory initialization

        Args:
            memory_size ([int]): [length of the experience Replay object]
            batch_size ([int]): [Sampling batch size]
        """
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    # Put data in policy replay memory
    def add(self, state:np.array, action:np.array, reward:list, next_state:np.array):
        """[Add Experience]

        Args:
            state ([Numpy Array]): [State]
            action ([Numpy Array]): [Action]
            reward ([Numpy Array]): [reward]
            next_s ([Numpy Array]): [next State]
        """
        
        if self.curr_lens == 0:
            self.states = state
            self.actions = action
            self.rewards = reward
            self.next_states = next_state
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, state), axis=0)
            self.next_states = np.concatenate((self.next_states, next_state), axis=0)
            self.actions = np.concatenate((self.actions, action), axis=0)
            self.rewards = np.concatenate((self.rewards, reward), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)
            self.states[index:(index + new_sample_lens)] = state
            self.actions[index:(index + new_sample_lens)] = action
            self.rewards[index:(index + new_sample_lens)] = reward
            self.next_states[index:(index + new_sample_lens)] = next_state

    # Take a batch of samples
    def sample(self)->list:
        """Returns a batch of experience

        Returns:
            [list]: [Batch of experience]
        """
        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, self.rewards, self.next_states]
        indices = random.sample(list(range(0, self.curr_lens)), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_mask = self.next_states[indices]
        return [batch_s, batch_a, batch_r, batch_mask]

    def reset(self)->None:
        """reset the variables
        """
        self.states = []
        self.actions = []

        self.rewards = []
        self.next_states = []
        self.curr_lens = 0
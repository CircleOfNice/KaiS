from gym import spaces


class NewSpace(spaces.Space):
    def __init__(self, shape):
        super(NewSpace, self).__init__(shape = shape)
    def contains(self,x):
        #print(x.shape, self.shape)
        if self.shape == x.shape:
            return True
        else: return False
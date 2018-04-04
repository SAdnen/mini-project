
class BaseAgent:

    """
    Parent abstract Agent.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, **kwargs):
        """ Returns random action from action space"""
        return self.action_space.sample()

    def learn(self, **kwargs):
        raise NotImplementedError

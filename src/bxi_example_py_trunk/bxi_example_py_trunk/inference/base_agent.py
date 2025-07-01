class baseAgent:
    def __init__(self,device):
        self.device = device
        self.last_actions_buf = None

    def build_observations(self,obs_group):
        raise NotImplementedError

    def inference(self,obs_group):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def last_actions(self):
        return self.last_actions_buf

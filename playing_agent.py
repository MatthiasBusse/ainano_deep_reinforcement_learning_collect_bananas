import torch
from dqn_model import QNetwork

class Playing_Agent(object):
    """An agent who chooses an action from experience
    """
    def __init__(self):
        self.qnetwork = QNetwork(state_size=37, action_size=4, seed=0)
        self.qnetwork.load_state_dict(torch.load('checkpoint.pth'))
        self.qnetwork.eval()

    def act(self, state):
        """Return experienced action given state"""

        state = torch.from_numpy(state).float().unsqueeze(0).to('cpu')
        with torch.no_grad():
            q = self.qnetwork(state)            # action values
        return q.argmax().item()

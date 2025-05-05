import numpy as np

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

import math
import torch
import torch.nn as nn

from wrappers import *

device = "cuda"

class model(nn.Module):
    def __init__(self, n_frame, n_action, noisy, device):
        super(model, self).__init__()
        self.noisy = noisy
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_frame, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Dynamically calculate the flattened size
        dummy_input = torch.zeros(1, n_frame, 84, 84)  # Example input size
        with torch.no_grad():
            dummy_output = self.conv_layers(dummy_input)
        flattened_size = dummy_output.numel()

        self.fc = nn.Linear(flattened_size, 512)
        self.q = nn.Linear(512, n_action)
        self.v = nn.Linear(512, 1)

        self.device = device

        # Initialize weights for convolutional layers
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        if self.noisy:
            self.reset_noise()
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv - adv.mean())

        return q

def normalize(x):
    return x.to(torch.float32) / 255.0


n_frame = 4
env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = wrap_mario(env)
q = model(n_frame, env.action_space.n, False, device).to(device)
q.load_state_dict(torch.load("../DRL-Assignment-3/model.pth", weights_only=True))


#frames = []
done = False
s = env.reset(force=True)
#q.eval()
score = 0
while not done:
    # frames.append(s[3])
    # frames.append(env.render(mode="rgb_array"))
    with torch.no_grad():
        s_tensor = torch.from_numpy(s).to(device)
        s_expanded = normalize(s_tensor).unsqueeze(0)
        a = q(s_expanded).argmax().item()
    s_next, r, done, _ = env.step(a)
    score += r
    s = s_next
#filepath = os.path.join(recordings_dir, f"{k}.gif")
#imageio.mimsave(filepath, frames)
print(f"Score {score}")# Saved gif to {filepath}")
#q.train()
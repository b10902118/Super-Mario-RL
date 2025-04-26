import pickle
import random
from collections import deque

import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import time

from wrappers import *

from torchrl.modules import NoisyLinear
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import tensorclass

torch.set_float32_matmul_precision("high")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(device)


def arrange(s):
    # if not type(s) == "numpy.ndarray":
    #    s = np.array(s)
    # (84, 84 ,4)
    # print("arrange(): ", s.shape)
    # print(type(s)) # numpy.ndarray
    # assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    ret = np.expand_dims(ret, 0)
    return ret
    # print("arrange(): ", ret.shape)
    return torch.tensor(ret, dtype=torch.float32)


class replay_memory(object):
    def __init__(self, N):
        self.memory = deque(maxlen=N)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, n):
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)


class model(nn.Module):
    def __init__(self, n_frame, n_action, device):
        super(model, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 3, 1)

        # Dynamically calculate the flattened size
        dummy_input = torch.zeros(1, n_frame, 84, 84)  # Example input size
        with torch.no_grad():
            dummy_output = self.layer2(self.layer1(dummy_input))
        flattened_size = dummy_output.numel()  # Total elements in the output

        # self.fc = nn.Linear(flattened_size, 512)
        # self.q = nn.Linear(512, n_action)
        # self.v = nn.Linear(512, 1)
        # Replace with NoisyLinear from torchrl
        self.fc = NoisyLinear(flattened_size, 512)
        self.q = NoisyLinear(512, n_action)
        self.v = NoisyLinear(512, 1)

        self.device = device

        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv - adv.mean())

        return q


def soft_update(q, q_target, tau=0.001):
    for param, target_param in zip(q.parameters(), q_target.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def train(q, q_target, memory, batch_size, gamma, optimizer):
    # print("train()")
    s, r, a, s_prime, done = list(map(list, zip(*memory.sample(batch_size))))
    s = np.array(s).squeeze()
    s_prime = np.array(s_prime).squeeze()
    s = torch.tensor(s, dtype=torch.float32).to(device)
    s_prime = torch.tensor(s_prime, dtype=torch.float32).to(device)
    a_max = q(s_prime).max(1)[1].unsqueeze(-1)
    r = torch.tensor(r, dtype=torch.float32).unsqueeze(-1).to(device)
    done = torch.tensor(done, dtype=torch.float32).unsqueeze(-1).to(device)
    with torch.no_grad():
        y = r + gamma * q_target(s_prime).gather(1, a_max) * (1 - done)
    a = torch.tensor(a).unsqueeze(-1).to(device)
    q_value = torch.gather(q(s), dim=1, index=a.view(-1, 1).long())

    loss = F.smooth_l1_loss(q_value, y).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # soft_update(q, q_target)
    return loss


def copy_weights(q, q_target):
    q_dict = q.state_dict()
    q_target.load_state_dict(q_dict)


def main(env, q, q_target, optimizer, scheduler):
    t = 0
    gamma = 0.99
    batch_size = 256

    N = 50000
    # eps = 0.001
    memory = replay_memory(N)
    # memory = ReplayBuffer(storage=LazyTensorStorage(N), pin_memory=True)
    update_interval = 50
    print_interval = 10

    score_lst = []
    total_score = 0.0
    loss = 0.0
    start_time = time.perf_counter()
    step_count = 0

    for k in range(1, 10000 + 1):
        s = arrange(env.reset())
        done = False

        while not done:
            # if eps > np.random.rand():
            # a = env.action_space.sample()
            # else:
            with torch.no_grad():
                a = q(torch.tensor(s, dtype=torch.float32).to(device)).argmax().item()
            s_prime, r, done, _ = env.step(a)
            s_prime = arrange(s_prime)
            total_score += r
            # reward shaping
            r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r
            memory.push((s, float(r), int(a), s_prime, done))
            s = s_prime
            stage = env.unwrapped._stage
            # print(f"{len(memory)}")
            if len(memory) > batch_size:
                loss += train(q, q_target, memory, batch_size, gamma, optimizer)
                t += 1
            if (t + 1) % update_interval == 0:
                copy_weights(q, q_target)
                torch.save(q.state_dict(), "mario_q.pth")
                torch.save(q_target.state_dict(), "mario_q_target.pth")
            step_count += 1
        # scheduler.step()

        if k % print_interval == 0:
            time_spent = time.perf_counter() - start_time
            # 20 for train, 130 for no train, 150 for pure random
            print(
                f"Epoch: {k} | Score: {total_score / print_interval:.6f} | "
                f"Loss: {loss / print_interval:.2f} | Stage: {stage} | Time Spent: {time_spent:.6f}| Speed: {step_count / time_spent} steps/s | Learning Rate: {scheduler.get_last_lr()[0]:.6f}"
            )
            score_lst.append(total_score / print_interval)
            total_score = 0
            loss = 0.0
            start_time = time.perf_counter()
            step_count = 0
            pickle.dump(score_lst, open("score.p", "wb"))


if __name__ == "__main__":
    n_frame = 4
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    # print(f"{env.action_space.n=}") # 12
    q = model(n_frame, env.action_space.n, device).to(device)
    q_target = model(n_frame, env.action_space.n, device).to(device)

    q.compile()
    q_target.compile()

    q.train()  # keep noisy layers in training mode
    q_target.eval()

    optimizer = optim.Adam(q.parameters(), lr=0.0002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    main(env, q, q_target, optimizer, scheduler)

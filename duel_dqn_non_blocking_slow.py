import pickle
import random
import numpy as np

import gym_super_mario_bros

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import time

from wrappers import *

from torchrl.modules import NoisyLinear

# from torchrl.data import ReplayBuffer, LazyTensorStorage, ListStorage
from tensordict import TensorDict
import imageio
from PIL import Image
import os
from datetime import datetime

torch.set_float32_matmul_precision("high")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(device)

# Get current time in MM_DD_HH_MM format
current_time = datetime.now().strftime("%m_%d_%H_%M")

# Create the folder under ./recordings
recordings_dir = os.path.join("recordings", current_time)
os.makedirs(recordings_dir, exist_ok=True)
print(f"Created directory: {recordings_dir}")


class ReplayBuffer:
    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_size)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, size=None, to_device=None):
        if size is None:
            size = self.batch_size
        if len(self.buffer) < size:
            raise ValueError("Not enough samples in the buffer to sample from.")
        batch = random.sample(self.buffer, size)
        # Convert to TensorDict
        batch_dict = {
            "s": torch.from_numpy(np.array([np.array(b["s"]) for b in batch])),
            "r": torch.tensor([b["r"] for b in batch], dtype=torch.float32),
            "a": torch.tensor([b["a"] for b in batch], dtype=torch.long),
            "s_prime": torch.from_numpy(
                np.array([np.array(b["s_prime"]) for b in batch])
            ),
            "done": torch.tensor([b["done"] for b in batch], dtype=torch.long),
        }

        ret = TensorDict(batch_dict, batch_size=size)
        if to_device is not None:
            ret.pin_memory()
            return ret.to(to_device, non_blocking=True)
        return ret

    def __len__(self):
        return len(self.buffer)


class model(nn.Module):
    def __init__(self, n_frame, n_action, device):
        super(model, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_frame, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 3, 1), nn.ReLU()
        )

        # Dynamically calculate the flattened size
        dummy_input = torch.zeros(1, n_frame, 84, 84)  # Example input size
        with torch.no_grad():
            dummy_output = self.conv_layers(dummy_input)
        flattened_size = dummy_output.numel()

        # self.fc = nn.Linear(flattened_size, 512)
        # self.q = nn.Linear(512, n_action)
        # self.v = nn.Linear(512, 1)
        # Replace with NoisyLinear from torchrl
        self.fc = NoisyLinear(flattened_size, 512)
        self.q = NoisyLinear(512, n_action)
        self.v = NoisyLinear(512, 1)

        self.device = device

        # Initialize weights for convolutional layers
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        self.reset_noise()
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv - adv.mean())

        return q

    def reset_noise(self):
        self.fc.reset_noise()
        self.q.reset_noise()
        self.v.reset_noise()


def soft_update(q, q_target, tau=0.001):
    for param, target_param in zip(q.parameters(), q_target.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def train(q, q_target, batch, batch_size, gamma, optimizer):
    # s, r, a, s_prime, done = list(map(list, zip(*memory.sample(batch_size))))
    s, r, a, s_prime, done = (
        batch["s"],
        batch["r"],
        batch["a"],
        batch["s_prime"],
        batch["done"],
    )

    # print(type(s), type(r), type(a), type(s_prime), type(done))
    # print(s.shape, r.shape, a.shape, s_prime.shape, done.shape)
    a_max = q(s_prime).max(1)[1].unsqueeze(-1)  # [1] is indices
    with torch.no_grad():
        y = r + gamma * q_target(s_prime).gather(1, a_max).squeeze(-1) * (1 - done)
    # q(s).shape = (batch_size, 12)
    # a.shape = (batch_size, 1)
    q_value = torch.gather(q(s), dim=1, index=a.unsqueeze_(-1)).squeeze(-1)
    # q_value.shape = (batch_size)

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
    # replay_buffer = ReplayBuffer(storage=ListStorage(N), batch_size=batch_size)
    replay_buffer = ReplayBuffer(max_size=N, batch_size=batch_size)
    update_interval = 50
    eval_episode = 100
    print_interval = 50

    total_score = 0.0
    loss = 0.0
    start_time = time.perf_counter()
    step_count = 0
    stage = 1

    for k in range(1, 5000 + 1):
        s = env.reset()  # observation is LazyFrame
        done = False
        batch = None

        while not done:
            with torch.no_grad():
                s_expanded = np.expand_dims(s, 0)
                a = q(torch.from_numpy(s_expanded).to(device)).argmax().item()
            s_prime, r, done, _ = env.step(a)
            total_score += r
            # reward shaping
            r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r
            # print(s.dtype) #float32

            # minimal speed drop
            replay_buffer.push(
                {
                    "s": s,
                    "r": r,
                    "a": a,
                    "s_prime": s_prime,
                    "done": done,
                }
            )
            # cpu_buffer.add(s, r, a, s_prime, done)
            # if cpu_buffer.is_full():
            #    replay_buffer.extend(cpu_buffer.flush())
            s = s_prime
            stage = max(stage, env.unwrapped._stage)
            # print(f"{len(memory)}")
            if batch is not None:
                loss += train(q, q_target, batch, batch_size, gamma, optimizer)
                t += 1
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(to_device=device)
            if (t + 1) % update_interval == 0:
                copy_weights(q, q_target)
            #    torch.save(q.state_dict(), "mario_q.pth")
            #    torch.save(q_target.state_dict(), "mario_q_target.pth")
            step_count += 1
            if step_count == 512:
                time_spent = time.perf_counter() - start_time
                print(f"Speed: {step_count / time_spent} steps/s")
                step_count = 0
                start_time = time.perf_counter()
        # scheduler.step()
        # soft_update(q, q_target, tau=0.01)

        if k % print_interval == 0:
            time_spent = time.perf_counter() - start_time
            # 20 for train, 130 for no train, 150 for pure random
            print(
                f"Epoch: {k} | Score: {total_score / print_interval:.6f} | "
                f"Loss: {loss / print_interval:.2f} | Stage: {stage} | Time Spent: {time_spent:.6f}| Speed: {step_count / time_spent} steps/s | Learning Rate: {scheduler.get_last_lr()[0]:.6f}"
            )
            total_score = 0
            loss = 0.0
            start_time = time.perf_counter()
            step_count = 0
            stage = 1

        if k % eval_episode == 0:
            frames = []
            done = False
            s = arrange(env.reset())
            q.eval()
            score = 0
            while not done:
                frames.append(s[3] * 255)
                # frames.append(env.render(mode="rgb_array"))
                with torch.no_grad():
                    s_expanded = np.expand_dims(s, 0)
                    a = (
                        q(torch.tensor(s_expanded, dtype=torch.float32).to(device))
                        .argmax()
                        .item()
                    )
                s_prime, r, done, _ = env.step(a)
                s_prime = arrange(s_prime)
                score += r
                s = s_prime
            # frames.append(env.render(mode="rgb_array"))
            filepath = os.path.join(recordings_dir, f"{k}.gif")
            # imagedir = os.path.join("recordings", f"{k}")
            # if os.path.exists(imagedir):
            #    for file in os.listdir(imagedir):
            #        os.remove(os.path.join(imagedir, file))
            # else:
            #    os.makedirs(imagedir, exist_ok=True)
            # frames = [Image.fromarray(frame) for frame in frames]
            # for i in range(len(frames)):
            #    frames[i].save(os.path.join(imagedir, f"{i}.jpg"))
            imageio.mimsave(filepath, frames)
            print(f"Score {score}, Saved gif to {filepath}")
            # print(f"Saved {len(frames)} frames to {imagedir}")
            q.train()


if __name__ == "__main__":
    n_frame = 4
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    # print(f"{env.action_space.n=}") # 12
    q = model(n_frame, env.action_space.n, device).to(device)
    q_target = model(n_frame, env.action_space.n, device).to(device)

    # q.compile()
    # q_target.compile()

    q.train()  # keep noisy layers in training mode
    q_target.eval()

    optimizer = optim.Adam(q.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    training_start_time = time.perf_counter()
    main(env, q, q_target, optimizer, scheduler)
    training_end_time = time.perf_counter()
    print(
        f"Total training time: {training_end_time - training_start_time:.2f} seconds"
    )  # Print the training time

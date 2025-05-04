import pickle
import random
import numpy as np

import gym_super_mario_bros

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import time

from wrappers import *

from torchrl.modules import NoisyLinear

from torchrl.data import ReplayBuffer, LazyTensorStorage, ListStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tensordict import TensorDict
import imageio
from PIL import Image
import os
from datetime import datetime
from copy import deepcopy

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

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9999  # 992
epsilon_base = 1


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
        print(f"Flattened size: {flattened_size}")  # 3136

        if noisy:
            self.fc = nn.Linear(flattened_size, 512)
            self.q = NoisyLinear(512, n_action)
            self.v = NoisyLinear(512, 1)
        else:
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

    def reset_noise(self):
        self.q.reset_noise()
        self.v.reset_noise()


def normalize(x):
    return x.to(torch.float32) / 255.0


def train(q, q_target, replay_buffer, prioritize, batch_size, gamma, optimizer):
    # s, r, a, s_prime, done = list(map(list, zip(*memory.sample(batch_size))))
    if prioritize:
        batch, info = replay_buffer.sample(return_info=True)
    else:
        batch = replay_buffer.sample()

    s, r, a, s_prime, done = (
        batch["s"],
        batch["r"],
        batch["a"],
        batch["s_prime"],
        batch["done"],
    )
    s = normalize(s)
    s_prime = normalize(s_prime)

    # print(type(s), type(r), type(a), type(s_prime), type(done))
    # print(s.shape, r.shape, a.shape, s_prime.shape, done.shape)
    a_max = q(s_prime).max(1)[1].unsqueeze(-1)  # [1] is indices
    with torch.no_grad():
        y = r + gamma * q_target(s_prime).gather(1, a_max).squeeze(-1) * (1 - done)
    # q(s).shape = (batch_size, 12)
    # a.shape = (batch_size, 1)
    q_value = torch.gather(q(s), dim=1, index=a.unsqueeze_(-1)).squeeze(-1)
    # q_value.shape = (batch_size)

    if prioritize:
        loss = (F.smooth_l1_loss(q_value, y) * info["_weight"].to(device)).mean()
    else:
        loss = F.smooth_l1_loss(q_value, y).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if prioritize:
        deltas = y - q_value
        replay_buffer.update_priority(index=info["index"], priority=deltas.abs())

    return loss


def soft_update(q, q_target, tau=0.001):
    for param, target_param in zip(q.parameters(), q_target.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(q, q_target):
    q_dict = q.state_dict()
    q_target.load_state_dict(q_dict)


def get_epsilon(cur_x, mean_x, std_x):
    global epsilon_base
    # decide epsilon
    # safety check
    if mean_x == 0:
        epsilon = 1
    # multi-stage linear
    elif cur_x < mean_x:
        epsilon = 0.04 * (cur_x / mean_x)

    # safety check
    elif std_x == 0:
        epsilon = 1
    elif mean_x + std_x > cur_x > mean_x:
        epsilon = 0.04 + (cur_x - mean_x) / std_x * 0.06
    else:
        epsilon = 0.1 + min((cur_x - mean_x - std_x), std_x) / std_x * 0.1

    epsilon += epsilon_base
    epsilon_base *= EPSILON_DECAY
    return min(1, epsilon)


def calc_x(x_pos, stage):
    # assume world 1
    return 2000 * (stage - 1) + x_pos


def main(
    env,
    q,
    q_target,
    optimizer,
    scheduler,
    episodes,
    bufsize,
    soft,
    no_prio,  # always
    epsilon_greedy,  # always
    gamma,
    alpha=0.6,
    beta=0.4,
    hard_update_interval=50,
):
    t = 0
    batch_size = 256

    prioritize = False
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(bufsize, device=device),
        batch_size=batch_size,
    )

    eval_interval = 100
    save_interval = 1000
    print_interval = 50

    total_score = 0.0
    loss = 0.0
    start_time = time.perf_counter()
    step_count = 0
    max_stage = 1
    big_count = 0
    green_count = 0

    last_400_x = deque([0], maxlen=400)
    max_score = -10000000
    min_score = -max_score
    epsilon = EPSILON_START

    for k in range(1, episodes + 1):
        s = env.reset()
        info = env.unwrapped._get_info()
        s = torch.from_numpy(s).to(device)
        done = False
        cur_score = 0
        # max_x = info["x_pos"]
        # max_r = 0
        # min_r = 0
        # prev_x_pos = 0
        # stay_count = 0
        small = True
        life = info["life"]

        while not done:
            cur_x = calc_x(info["x_pos"], info["stage"])
            mean_x = np.mean(last_400_x)
            std_x = np.std(last_400_x)
            epsilon = get_epsilon(cur_x, mean_x, std_x)
            if random.random() < epsilon:
                # a = random.randint(0, env.action_space.n - 1)
                a = random.choice([0, 1, 2, 3, 4, 5, 6])  # simple actions
            else:
                with torch.no_grad():
                    s_expanded = normalize(s).unsqueeze(0)
                    a = q(s_expanded).argmax().item()

            s_next, r, done, info = env.step(a)
            s_next = torch.from_numpy(s_next).to(device)
            # {'coins': 2, 'flag_get': False, 'life': 0, 'score': 400, 'stage': 1, 'status': 'small', 'time': 398, 'world': 1, 'x_pos': 99, 'y_pos': 79}
            # r [-15, 15]
            total_score += r
            cur_score += r
            # max_x = max(max_x, info["x_pos"])
            # max_r = max(max_r, r)
            # min_r = min(min_r, r)

            # reward shaping
            # r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r
            if small and info["status"] != "small":
                r += 30
                small = False
                big_count += 1
            elif not small and info["status"] == "small":
                # should be handled by env
                # r -= 15
                small = True

            if info["life"] > life and info["life"] != 255:
                r += 120
                life = info["life"]
                green_count += 1

            r /= 15

            # r [-3.015, 3.015]
            # print(s.dtype) #float32

            # minimal speed drop (71 -> 68), TensorDict no gain
            replay_buffer.add(
                {
                    "s": s,
                    "r": torch.tensor(r, dtype=torch.float32),
                    "a": torch.tensor(a, dtype=torch.long),  # indexing
                    "s_prime": s_next,
                    "done": torch.tensor(done, dtype=torch.int32),
                }
            )
            s = s_next
            # TODO: increase epsilon for new stage
            cur_stage = env.unwrapped._stage
            max_stage = max(max_stage, cur_stage)
            if len(replay_buffer) > batch_size:
                loss += train(
                    q, q_target, replay_buffer, prioritize, batch_size, gamma, optimizer
                )
                t += 1

            if soft:
                soft_update(q, q_target, tau=0.01)
            elif (t + 1) % hard_update_interval == 0:
                hard_update(q, q_target)

            step_count += 1

            # if prev_x_pos == info["x_pos"]:
            #    stay_count += 1
            #    if stay_count > 15:
            #        break
            # else:
            #    stay_count = 0
            #    prev_x_pos = info["x_pos"]

            # if step_count == 512:
            #    time_spent = time.perf_counter() - start_time
            #    print(f"Speed: {step_count / time_spent} steps/s")
            #    step_count = 0
            #    start_time = time.perf_counter()
        # print(
        #    f"episode {k} result: ",
        #    f"{max_x=}",
        #    f"{max_r=}",
        #    f"{min_r=}",
        #    f"score: {cur_score}",
        # )

        last_400_x.append(cur_x)
        max_score = max(max_score, cur_score)
        min_score = min(min_score, cur_score)
        # prev_mean_score = mean_score
        # mean_score = ((k - 1) * mean_score + cur_score) / k
        # prev_Ex2 = std_score**2 + prev_mean_score**2
        # Ex2 = ((k - 1) * prev_Ex2 + cur_score**2) / k
        # std_score = math.sqrt(Ex2 - mean_score**2)

        if k % print_interval == 0:
            time_spent = time.perf_counter() - start_time
            # 20 for train, 130 for no train, 150 for pure random
            print(
                f"Epoch: {k} | Score: {total_score / print_interval:.2f} | "
                f"Loss: {loss / print_interval:.2f} | Stage: {max_stage} | Time Spent: {time_spent:.1f}| Speed: {step_count / time_spent:.1f} steps/s |"
                f" Max: {max_score}  Min: {min_score}  Mean_x: {mean_x:.2f}  Std_x: {std_x:.2f} | Big: {big_count} Green: {green_count}"
            )
            max_score = -10000000
            min_score = -max_score
            total_score = 0
            loss = 0.0
            start_time = time.perf_counter()
            step_count = 0
            max_stage = 1
            big_count = 0
            green_count = 0

        # TODO: run three lives
        if k % eval_interval == 0:
            frames = []
            done = False
            s = env.reset(force=True)
            q.eval()
            score = 0
            while not done:
                frames.append(s[3])
                # frames.append(env.render(mode="rgb_array"))
                with torch.no_grad():
                    s_tensor = torch.from_numpy(s).to(device)
                    s_expanded = normalize(s_tensor).unsqueeze(0)
                    a = q(s_expanded).argmax().item()
                s_next, r, done, _ = env.step(a)
                score += r
                s = s_next
            filepath = os.path.join(recordings_dir, f"{k}.gif")
            imageio.mimsave(filepath, frames)
            print(f"Score {score}, Saved gif to {filepath}")
            q.train()
        if k % save_interval == 0:
            torch.save(q.state_dict(), os.path.join(recordings_dir, "mario_q.pth"))


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Dueling DQN on Super Mario Bros."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--episodes", type=int, default=20000, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--bufsize", type=int, default=100000, help="Buffer size for replay buffer"
    )
    parser.add_argument(
        "--soft",
        action="store_true",
        default=False,
        help="Use soft update for target network",
    )
    parser.add_argument(
        "--no-prio",
        action="store_true",
        default=True,
        help="Use prioritized replay buffer",
    )
    parser.add_argument(
        "--eps",
        action="store_true",
        default=True,
        help="Use noisy layers for exploration",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.63,
        help="Discount factor for the Q-learning update",
    )
    args = parser.parse_args()
    args_dict = vars(args)
    print(args_dict)

    n_frame = 4
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    # env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    # print(f"{env.action_space.n=}") # 12
    q = model(n_frame, env.action_space.n, not args.eps, device).to(device)
    q_target = model(n_frame, env.action_space.n, not args.eps, device).to(device)

    # q.compile()
    # q_target.compile()

    q.train()  # keep noisy layers in training mode
    q_target.eval()

    optimizer = optim.Adam(q.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    args_dict["epsilon_greedy"] = args_dict.pop("eps")
    args_dict.pop("lr")

    training_start_time = time.perf_counter()
    main(env, q, q_target, optimizer, scheduler, **args_dict)
    training_end_time = time.perf_counter()
    print(
        f"Total training time: {training_end_time - training_start_time:.2f} seconds"
    )  # Print the training time

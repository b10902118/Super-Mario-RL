import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import sys
import tty
import termios


def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


# env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = gym_super_mario_bros.make("SuperMarioBrosRandomStages-v0", stages=["1-2"])
env = JoypadSpace(env, SIMPLE_MOVEMENT)


env.reset()
env.render()
# print(env.unwrapped._get_info())
done = False
score = 0
print(SIMPLE_MOVEMENT)
while not done:
    action = getch()  # Read one character
    if action.isdigit():  # Ensure the input is a valid number
        action = int(action)
    else:
        action = 0
    total_reward = 0
    for i in range(4):
        _, reward, done, info = env.step(action)
        total_reward += reward
        if info["life"] < 2:
            break
    print()
    print(info)
    print(total_reward)
    score += total_reward
    if info["life"] < 2:
        print("Game Over")
        break
    env.render()
print("Score: ", score)

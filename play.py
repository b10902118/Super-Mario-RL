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

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)


env.reset()
env.render()
done = False
score = 0
print(SIMPLE_MOVEMENT)
while not done:
    action = getch()  # Read one character
    if action.isdigit():  # Ensure the input is a valid number
        action = int(action)
    else:
        action = 0
    for i in range(4):
        _, reward, done, _ = env.step(action)
        if done:
            break
    score += reward
    env.render()
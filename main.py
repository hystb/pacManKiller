import cv2
from train import *
from DQNCNN import DQNCNN
import gym

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
env = gym.make('MsPacman-v4')

num_episodes = 1000
batch_size = 32
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
learning_rate = 0.0001


action_space = env.action_space.n
print(action_space)
model = DQNCNN(action_space)

rewards = train(model, env, num_episodes, batch_size, gamma, epsilon, epsilon_decay, min_epsilon, learning_rate)

from collections import deque
from preprocessData import *
import torch
import torch.optim as optim
import torch.nn as nn
import random

import cv2
import numpy as np
import time
window_size = (800, 600)

preprocess = PreprocessFrame()

def train(model, env, num_episodes, batch_size, gamma, epsilon, epsilon_decay, min_epsilon, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    replay_buffer = deque(maxlen=10000)
    all_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        stacked_frames = deque([preprocess(state) for _ in range(4)], maxlen=4)
        stacked_state, _ = stack_frames(stacked_frames, state, True, preprocess)
        total_reward = 0

        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(stacked_state)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            frame = env.render(mode='rgb_array')
            total_reward += reward
            next_stacked_state, stacked_frames = stack_frames(stacked_frames, next_state, False, preprocess)
            
            replay_buffer.append((stacked_state, action, reward, next_stacked_state, done))
            stacked_state = next_stacked_state 
            
            resized_frame = cv2.resize(frame, window_size)
            cv2.imshow('Ms. Pacman', resized_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states)
                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.cat(next_states)
                dones = torch.tensor(dones, dtype=torch.float32)

                current_q_values = model(states).gather(1, actions.view(-1, 1)).squeeze()
                next_q_values = model(next_states).max(1)[0]
                expected_q_values = rewards + (gamma * next_q_values * (1 - dones))

                loss = loss_fn(current_q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                epsilon = max(min_epsilon, epsilon * epsilon_decay)
                break

        all_rewards.append(total_reward)
        print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {epsilon}")
    cv2.destroyAllWindows()
    return all_rewards



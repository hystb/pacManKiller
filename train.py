from collections import deque
from datetime import datetime
from preprocessData import *
import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np

# import cv2
# import numpy as np
# import time
# window_size = (800, 600)

def train(model, env, num_episodes, batch_size, gamma, epsilon, epsilon_decay, min_epsilon, learning_rate, device):
    model.to(device)
    max_reward = 0
    count_frame = 1
    normalization_factor = 100000
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    replay_buffer = deque(maxlen=10000)
    all_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        stacked_frames = deque([preprocess(state) for _ in range(4)], maxlen=4)
        stacked_state, _ = stack_frames(stacked_frames, state, True, preprocess)
        stacked_state = stacked_state.to(device) 
        total_reward = 0

        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                model.eval()
                with torch.no_grad():
                    q_values = model(stacked_state)
                    action = torch.argmax(q_values).item()
                model.train()

            next_state, reward, done, info = env.step(action)
            
            reward = reward / normalization_factor
            
            total_reward += reward
            
            next_stacked_state, stacked_frames = stack_frames(stacked_frames, next_state, False, preprocess)
            
            replay_buffer.append((stacked_state, action, reward, next_stacked_state, done))
            stacked_state = next_stacked_state.to(device)

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.stack(states).to(device).squeeze(1)
                actions = torch.tensor(actions).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.stack(next_states).to(device).squeeze(1)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = model(next_states).max(1)[0]
                expected_q_values = rewards + (gamma * next_q_values * (1 - dones))

                loss = loss_fn(current_q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            count_frame += 1
            if done or count_frame > 2000000:
                epsilon = max(min_epsilon, epsilon * epsilon_decay)
                count_frame = 0
                break

        all_rewards.append(total_reward)
        print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {epsilon}")
        if total_reward > max_reward:
            max_reward = total_reward
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = f"modelmax{total_reward*normalization_factor}_{timestamp}.pth"
            torch.save(model.state_dict(), model_save_path)
        if episode % 1000 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = f"model_{timestamp}.pth"
            torch.save(model.state_dict(), model_save_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"model_{timestamp}.pth"
    torch.save(model.state_dict(), model_save_path)
    # cv2.destroyAllWindows()
    return all_rewards, model_save_path



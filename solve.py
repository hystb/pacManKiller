import torch
from preprocessData import *

import cv2
import numpy as np
import time
window_size = (800, 600)

def load_model(model, model_load_path, device):
    model.load_state_dict(torch.load(model_load_path))
    model.to(device)
    model.eval()
    return model

def predict(model, state, device):
    state = state.to(device)
    with torch.no_grad():
        q_values = model(state)
        action = torch.argmax(q_values).item()
    return action

def simulate_game(model, env, device):
    state = env.reset()
    stacked_frames = deque([preprocess(state) for _ in range(4)], maxlen=4)
    stacked_state, _ = stack_frames(stacked_frames, state, True, preprocess)
    stacked_state = stacked_state.to(device)
    total_reward = 0

    while True:
        frame = env.render(mode='rgb_array')
        action = predict(model, stacked_state, device)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        next_stacked_state, stacked_frames = stack_frames(stacked_frames, next_state, False, preprocess)
        next_stacked_state = next_stacked_state.to(device)
        stacked_state = next_stacked_state
        
        resized_frame = cv2.resize(frame, window_size)  
        cv2.imshow('Ms. Pacman', resized_frame)
        if cv2.waitKey(int(1000/60)) & 0xFF == ord('q'):
            break
        if done:
            break

    env.close()
    print(f"Total Reward: {total_reward}")
    return total_reward

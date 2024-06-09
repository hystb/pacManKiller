import argparse
import torch
import gym
from solve import load_model, simulate_game
from train import train 
from DQNCNN import DQNCNN
import cv2

def main():
    parser = argparse.ArgumentParser(description="Train or execute a DQN model for Ms. Pacman")
    parser.add_argument('--mode', choices=['train', 'execute'], required=True, help="Choose whether to train a new model or execute an existing one")
    parser.add_argument('--path', type=str, help="Path to the pre-trained model (required for execution mode)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    env = gym.make('MsPacman-v4')

    if args.mode == 'train':
        num_episodes = 2000000
        batch_size = 64
        gamma = 0.99
        epsilon = 1.0
        epsilon_decay = 0.9999
        min_epsilon = 0.01
        learning_rate = 0.0001

        action_space = env.action_space.n
        model = DQNCNN(action_space)
        if args.path:
            print("Model Loaded")
            model = load_model(model, args.path, device)
            epsilon = 0.01
        rewards, path = train(model, env, num_episodes, batch_size, gamma, epsilon, epsilon_decay, min_epsilon, learning_rate, device)
        print(f"Training complete. Model saved to {path}")
        simulate_game(model, env, device)
    elif args.mode == 'execute':
        if not args.path:
            print("Error: --path is required for execution mode")
            return
        
        action_space = env.action_space.n
        model = DQNCNN(action_space)

        model = load_model(model, args.path, device)
        simulate_game(model, env, device)

if __name__ == "__main__":
    main()

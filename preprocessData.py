from torchvision import transforms
from collections import deque
import torch

class PreprocessFrame:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])

    def __call__(self, frame):
        frame = self.transform(frame)
        return frame

def stack_frames(stacked_frames, frame, is_new_episode, preprocess):
    frame = preprocess(frame)
    if is_new_episode:
        stacked_frames = deque([frame for _ in range(4)], maxlen=4)
    else:
        stacked_frames.append(frame)
    stacked_state = torch.cat(list(stacked_frames), dim=0).unsqueeze(0)
    return stacked_state, stacked_frames


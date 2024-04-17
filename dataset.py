import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video

class VideoDataset(Dataset):
    def __init__(self, dataset, dataset_type, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset[idx]

        video, _, _ = read_video(path, pts_unit='sec', output_format='TCHW')

        video = video.float() / 255.0

        if len(video) > 48:
            video = video[:48]

        if self.transform:
            video = torch.stack([self.transform(frame) for frame in video])

        video = video.permute(1, 0, 2, 3)
        return video, label


def get_dataset_paths(root_dir, labels=False):
    classes = os.listdir(root_dir)

    if labels:
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        idx_to_class = {i: classes[i] for i in range(len(classes))}

    dataset = []
    for cls in classes:
        class_dir = os.path.join(root_dir, cls)
        for clip in os.listdir(class_dir):

            if clip.endswith('.png'):
                continue

            clip_path = os.path.join(class_dir, clip)

            if labels:
                dataset.append((clip_path, class_to_idx[cls]))
            else:
                dataset.append(clip_path)

    if labels:
        return dataset, class_to_idx, idx_to_class
    else:
        return dataset
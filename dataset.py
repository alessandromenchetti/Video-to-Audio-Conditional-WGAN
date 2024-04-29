import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, dataset, transform=None):
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


class AVDataset(Dataset):
    def __init__(self, video_paths, transform=None):
        self.video_paths = video_paths
        self.transform = transform

        with open('spec_range.txt', 'r') as f:
            lines = f.readlines()
            self.spec_max = float(lines[0].strip())
            self.spec_min = float(lines[1].strip())

        self.dataset = self._create_dataset()

    def _create_dataset(self):
        dataset = []
        for video_path in self.video_paths:
            spec_path = video_path.rsplit('.', 1)[0] + '.png'

            spec = Image.open(spec_path).convert('L')
            spec = torch.from_numpy(np.array(spec)).float()

            spec = spec / 255.0
            spec = spec * (self.spec_max - self.spec_min) + self.spec_min
            spec = 2 * (spec - self.spec_min) / (self.spec_max - self.spec_min) - 1
            spec = spec.unsqueeze(0)

            dataset.append((video_path, spec))

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, spec = self.dataset[idx]

        video, _, _ = read_video(path, pts_unit='sec', output_format='TCHW')

        video = video.float() / 255.0

        if len(video) > 48:
            video = video[:48]

        if self.transform:
            video = torch.stack([self.transform(frame) for frame in video])

        video = video.permute(1, 0, 2, 3)
        video = video.clone()
        spec = spec.clone()

        return video, spec, path

def custom_collate_AV(batch):
    try:
        videos = []
        specs = []
        paths = []

        for item in batch:
            videos.append(item[0])
            specs.append(item[1])
            paths.append(item[2])

        videos = torch.stack(videos, dim=0)
        specs = torch.stack(specs, dim=0)

        return videos, specs, paths

    except Exception as e:
        print('Error processing batch: ', e)
        for item in batch:
            print(item[0].shape, item[1].shape, item[2])
        raise e

def get_dataset_paths(root_dir, single_class=False, labels=False):

    if single_class:
        classes = [single_class]
    else:
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
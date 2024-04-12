from torch.utils.data import Dataset
from torchvision.io import read_video

class VideoDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset[idx]

        print('Reading video')
        video, _, _ = read_video(path, pts_unit='sec', output_format='TCHW')

        video = video.float() / 255.0

        if len(video) > 64:
            video = video[:64]

        print('Video read')
        return video, label
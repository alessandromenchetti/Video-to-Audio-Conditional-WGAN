import numpy as np
from scipy.linalg import sqrtm
import os

import torch
from torchaudio.transforms import InverseMelScale, GriffinLim

import openl3
import soundfile as sf

from dataset import AVDataset, get_dataset_paths
from sklearn.model_selection import train_test_split
from moviepy.editor import VideoFileClip
import torchaudio

def main():
    compute_L2_on_specs('test_set/specs_npy/')

    process_set_compute_fad('test_set/wavs/', '_orig.wav', '_fake.wav')
    process_set_compute_fad('test_set/wavs/', '_orig.wav', '_real.wav')

def load_embeddings(paths):
    embeddings = []
    for path in paths:
        audio, sr = sf.read(path)
        emb, _ = openl3.get_audio_embedding(audio, sr, content_type='env', input_repr='mel256', embedding_size=512, frontend='librosa')
        embeddings.append(emb.mean(axis=0))
    return np.array(embeddings)


def calc_fad(embeddings1, embeddings2):
    mu1, sigma1 = embeddings1.mean(axis=0), np.cov(embeddings1, rowvar=False)
    mu2, sigma2 = embeddings2.mean(axis=0), np.cov(embeddings2, rowvar=False)

    mu_diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    covmean = covmean.real
    return np.trace(sigma1 + sigma2 - 2 * covmean) + np.dot(mu_diff, mu_diff)


def process_set_compute_fad(dir, suffix1, suffix2):
    set1 = [os.path.join(dir, f) for f in os.listdir(dir) if suffix1 in f]
    set2 = [os.path.join(dir, f) for f in os.listdir(dir) if suffix2 in f]

    embeddings1 = load_embeddings(set1)
    embeddings2 = load_embeddings(set2)

    fad = calc_fad(embeddings1, embeddings2)
    print(f"FAD between {suffix1} and {suffix2}: {fad}")

def compute_L2_on_specs(dir):
    real_files = [os.path.join(dir, f) for f in os.listdir(dir) if '_real.npy' in f]
    fake_files = [os.path.join(dir, f) for f in os.listdir(dir) if '_fake.npy' in f]

    distances = []
    for real_file in real_files:
        base_name = real_file.split('_real')[0]
        fake_file = base_name + '_fake.npy'

        real_spec = np.load(real_file)
        fake_spec = np.load(fake_file)

        dist = np.linalg.norm(real_spec - fake_spec)
        distances.append(dist)

    avg_dist = np.mean(distances)
    print(f"Average L2 distance between real and fake spectograms: {avg_dist}")


def spec_to_waveform(spec_tensor):
    spec_tensor = (spec_tensor + 1) / 2

    with open('spec_range.txt', 'r') as f:
        lines = f.readlines()
        spec_max = float(lines[0].strip())
        spec_min = float(lines[1].strip())

    spec_tensor = spec_tensor * (spec_max - spec_min) + spec_min

    spec_tensor = torch.pow(2.0, spec_tensor)

    inverse_mel_scale = InverseMelScale(n_stft=(2048 // 2) + 1, n_mels=128, sample_rate=22050)
    griffin_lim = GriffinLim(n_fft=2048, n_iter=64, hop_length=512)

    stft = inverse_mel_scale(spec_tensor)
    waveform = griffin_lim(stft)

    waveform = waveform / waveform.abs().max()

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    return waveform

def save_wavs():
    firework_dataset = get_dataset_paths('data', single_class='fireworks')

    _, test_data = train_test_split(firework_dataset, test_size=0.1, random_state=42)
    test_dataset = AVDataset(test_data)

    test_paths = test_dataset.video_paths

    output_dir = 'test_set/wavs/'

    for path in test_paths:
        clip = VideoFileClip(path)
        clip.audio.write_audiofile(output_dir + path.split('\\')[-1].split('.')[0] + '_orig.wav')
        clip.close()
        print(f"Processed {path}")

    # Now load real and fake spectograms and convert them to a waveform and save in test_set/wavs as a .wav file
    for file in os.listdir('test_set/specs_npy/'):
        spec = np.load('test_set/specs_npy/' + file)
        spec_tensor = torch.from_numpy(spec)
        waveform = spec_to_waveform(spec_tensor)

        torchaudio.save(output_dir + file.split('.')[0] + '.wav', waveform, 22050)

        print(f"Processed {file}")


if __name__ == '__main__':
    main()
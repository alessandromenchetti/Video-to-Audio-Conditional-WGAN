import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import grad
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import torchaudio
from torchaudio.transforms import InverseMelScale, GriffinLim
from PIL import Image
import ffmpeg

from dataset import AVDataset, custom_collate_AV, get_dataset_paths
from gan_models import VideoEncoder, AudioGenerator, AudioCritic

def main():

    train_mode = False
    version = 9
    preload = 'models/audio_gan_v9_e400.pth'

    firework_dataset = get_dataset_paths('data', single_class='fireworks')

    train_data, test_data = train_test_split(firework_dataset, test_size=0.1, random_state=42)
    train_dataset = AVDataset(train_data)
    test_dataset = AVDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=4, collate_fn=custom_collate_AV, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False, num_workers=4, collate_fn=custom_collate_AV, pin_memory=True)

    video_encoder = VideoEncoder(pretrained_path='models/video_classifierResRNN_v3_e60.pth')
    generator = AudioGenerator()
    critic = AudioCritic()

    optimizer_G = Adam(generator.parameters(), lr=0.00002, betas=(0.0, 0.9))
    optimizer_C = Adam(critic.parameters(), lr=0.00001, betas=(0.0, 0.9))
    scheduler_G = lr_scheduler.ExponentialLR(optimizer_G, gamma=0.999)
    scheduler_C = lr_scheduler.ExponentialLR(optimizer_C, gamma=0.999)

    start_epoch = 0
    LAMBDA_GP = 15
    LAMBDA_FM = 10

    if preload:
        model_and_optimizer = torch.load(preload)
        generator.load_state_dict(model_and_optimizer['generator_state_dict'])
        critic.load_state_dict(model_and_optimizer['critic_state_dict'])
        optimizer_G.load_state_dict(model_and_optimizer['optimizer_G_state_dict'])
        optimizer_C.load_state_dict(model_and_optimizer['optimizer_C_state_dict'])
        scheduler_G.load_state_dict(model_and_optimizer['scheduler_G_state_dict']),
        scheduler_C.load_state_dict(model_and_optimizer['scheduler_C_state_dict']),
        start_epoch = model_and_optimizer['epoch']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if train_mode:
        train_gan(video_encoder, generator, critic, 400, start_epoch, train_loader, device, optimizer_G, optimizer_C, scheduler_G, scheduler_C, version, LAMBDA_GP, LAMBDA_FM)
    else:
        test_gan(video_encoder, generator, test_loader, device, version)


def calc_gp(critic, real_samples, fake_samples, video_encodings_expanded, device):
    batch_size, _, height, width = real_samples.size()

    alpha = torch.rand(batch_size, 1, height, width, device=device)
    alpha = alpha.expand_as(real_samples)

    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)

    interpolated_w_encodings = torch.cat((interpolated, video_encodings_expanded), dim=1)

    interpolated_scores = critic(interpolated_w_encodings)

    gradients = grad(outputs=interpolated_scores, inputs=interpolated_w_encodings,
                     grad_outputs=torch.ones(interpolated_scores.size(), device=device),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]


    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def ensure_optimizer_on_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def train_gan(video_encoder, generator, critic, n_epochs, start_epoch, train_loader, device, optimizer_G, optimizer_C, scheduler_G, scheduler_C, v, LAMBDA_GP=10, LAMBDA_FM=1):
    critic_losses = []
    gen_losses = []

    video_encoder.to(device)
    generator.to(device)
    critic.to(device)

    ensure_optimizer_on_device(optimizer_G, device)
    ensure_optimizer_on_device(optimizer_C, device)

    generator.train()
    critic.train()
    print('Starting training')
    for epoch in range(start_epoch, n_epochs):
        last_fake_specs, last_paths = None, None
        start_time = time.time()

        for i, data in enumerate(train_loader):
            videos, real_specs, paths = data
            videos, real_specs = videos.to(device), real_specs.to(device)

            noise = torch.randn(videos.size(0), 128, device=device)

            # Train Critic
            optimizer_C.zero_grad()

            video_encodings = video_encoder(videos)
            fake_specs = generator(torch.cat((video_encodings, noise), dim=1))

            video_encodings_expanded = video_encodings.unsqueeze(2).expand(-1, -1, 128).unsqueeze(1).transpose(2, 3)

            real_scores, real_features = critic(torch.cat((real_specs, video_encodings_expanded), dim=1), return_intermediate=True)
            fake_scores, fake_features = critic(torch.cat((fake_specs.detach(), video_encodings_expanded), dim=1), return_intermediate=True)
            critic_loss = -torch.mean(real_scores) + torch.mean(fake_scores)

            gp = calc_gp(critic, real_specs, fake_specs, video_encodings_expanded, device)

            total_critic_loss = critic_loss + LAMBDA_GP * gp

            total_critic_loss.backward()
            optimizer_C.step()

            # Training Generator
            if i % 5 == 0:
                optimizer_G.zero_grad()

                gen_specs = generator(torch.cat((video_encodings, noise), dim=1))
                gen_scores, gen_features = critic(torch.cat((gen_specs, video_encodings_expanded), dim=1), return_intermediate=True)

                gen_loss = -torch.mean(gen_scores)

                fm = mse_loss(gen_features, real_features.detach())
                total_gen_loss = gen_loss + LAMBDA_FM * fm

                total_gen_loss.backward()
                optimizer_G.step()

            last_fake_specs = fake_specs.detach().cpu()
            last_paths = paths

            if i % 10 == 0:
                print(f'Epoch {epoch + 1}, Iteration {i}, Critic Loss: {total_critic_loss.item()}, Gen Loss: {total_gen_loss.item()}, Time per 10 iterations: {time.time() - start_time}')

                critic_losses.append((epoch + i / len(train_loader), total_critic_loss.item()))
                gen_losses.append((epoch + i / len(train_loader), total_gen_loss.item()))

                start_time = time.time()

        scheduler_G.step()
        scheduler_C.step()

        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'generator_state_dict': generator.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_C_state_dict': optimizer_C.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_C_state_dict': scheduler_C.state_dict(),
                'epoch': epoch + 1
            }
            model_path = f'models/audio_gan_v{v}'
            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)
            output_name = f'audio_gan_v{v}_e{epoch + 1}.pth'

            try:
                torch.save(checkpoint, os.path.join(model_path, output_name))
                print(f'Checkpoint saved for epoch {epoch + 1}')
            except Exception as e:
                print(f'Error saving checkpoint for epoch {epoch + 1}: {e}')

            plot_output_name = output_name.rsplit('.', 1)[0] + '_losses.png'
            plot_losses(critic_losses, gen_losses, os.path.join(model_path, plot_output_name))

            test_example = last_fake_specs[0].float()

            out_path = last_paths[0].rsplit('.', 1)[0].rsplit('\\', 1)[1]

            save_spec(test_example, out_path, epoch + 1, v)
            waveform = spec_to_waveform(test_example)
            merge_audio_video(waveform, last_paths[0], epoch + 1, v)


def test_gan(video_encoder, generator, test_loader, device, v):
    video_encoder.to(device)
    generator.to(device)

    video_encoder.eval()
    generator.eval()

    for i, data in enumerate(test_loader):
        videos, real_specs, paths = data
        videos, real_specs = videos.to(device), real_specs.to(device)

        noise = torch.randn(videos.size(0), 128, device=device)

        with torch.no_grad():
            video_encodings = video_encoder(videos)
            fake_specs = generator(torch.cat((video_encodings, noise), dim=1))

        for j in range(len(paths)):
            path = paths[j]
            out_path = path.rsplit('.', 1)[0].rsplit('\\', 1)[1]

            real_spec = real_specs[j].cpu()
            save_spec(real_spec, out_path, 'real', 'test_set')

            fake_spec = fake_specs[j].cpu()
            save_spec(fake_spec, out_path, 'fake', 'test_set')

            waveform = spec_to_waveform(fake_spec)
            merge_audio_video(waveform, path, 'fake', 'test_set')


def plot_losses(critic_losses, gen_losses, output_path):
    critic_epochs, critic_vals = zip(*critic_losses)
    gen_epochs, gen_vals = zip(*gen_losses)

    plt.figure(figsize=(10, 5))
    plt.plot(critic_epochs, critic_vals, label='Critic Loss')
    plt.plot(gen_epochs, gen_vals, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def save_spec(spec_tensor, output_path, batch_idx, v):
    spec_tensor = spec_tensor.float()
    spec_tensor_01 = (spec_tensor + 1) / 2

    img = Image.fromarray((spec_tensor_01.squeeze(0).numpy() * 255).astype('uint8'))

    if v == 'test_set':
        output_dir = 'test_set/specs'
        output_dir_raw = 'test_set/specs_npy'
        np.save(os.path.join(output_dir_raw, f'{output_path}_{batch_idx}.npy'), spec_tensor.numpy())
    else:
        output_dir = f'train_v{v}'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    img.save(os.path.join(output_dir, f'{output_path}_{batch_idx}.png'))


def spec_to_waveform(spec_tensor):
    spec_tensor = spec_tensor.float()
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


def merge_audio_video(waveform, video_path, batch_idx, v):
    tmp_audio_path = os.path.join('temp', 'temp_audio.wav')

    output_path = video_path.rsplit('.', 1)[0].rsplit('\\', 1)[1]

    if v == 'test_set':
        output_dir = 'test_set/videos'
    else:
        output_dir = f'train_v{v}'

    torchaudio.save(tmp_audio_path, waveform, 22050)

    try:
        input_video = ffmpeg.input(video_path)
        input_audio = ffmpeg.input(tmp_audio_path)

        (
            ffmpeg
            .concat(input_video, input_audio, v=1, a=1)
            .output(f'{output_dir}/{output_path}_{batch_idx}.mp4', vcodec='libx264', acodec='aac', strict='experimental')
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print('FFmpeg Error:', e.stderr.decode())
        raise
    finally:
        os.remove(tmp_audio_path)


if __name__ == '__main__':
    main()
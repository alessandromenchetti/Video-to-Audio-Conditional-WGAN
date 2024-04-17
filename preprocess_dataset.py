import os
import ffmpeg
from concurrent.futures import ProcessPoolExecutor
import av
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from PIL import Image
import io

def get_video_info(video_path):
    """Return the frame rate of the video using PyAV."""
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            frame_rate = stream.average_rate
            duration = container.duration / av.time_base
            return frame_rate, duration
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

def save_spectogram(spec, filename, spec_max, spec_min):

    spec = (spec - spec_min) / (spec_max - spec_min)

    img = Image.fromarray((spec.numpy() * 255).astype('uint8'))
    img.save(filename)

def adjust_video_properties(input_path, output_path, start_time, end_time, spec_max, spec_min, target_fps=16, audio_sampling_rate=22050, target_width=224, target_height=224):
    """
    Adjusts the video frame rate to target_fps, converts audio to 22050 Hz mono,
    and resizes the video to fit the target height with padding to match the target width.
    """
    try:
        stream = ffmpeg.input(input_path, ss=start_time, to=end_time)

        video_stream = stream.video.filter('fps', fps=target_fps).filter('scale', -2, target_height)
        video_stream = video_stream.filter('crop', f'min(iw,{target_width})', target_height, '0', '0')
        video_stream = video_stream.filter('pad', target_width, target_height, '(ow-iw)/2', '(oh-ih)/2')

        audio_stream = stream.audio.filter('aformat', sample_rates=audio_sampling_rate, channel_layouts='mono')

        ffmpeg.output(video_stream, audio_stream, output_path, vcodec='libx264', acodec='aac').run(overwrite_output=True)

        audio, _ = ffmpeg.output(audio_stream, 'pipe:', format='wav').run(capture_stdout=True, capture_stderr=True, overwrite_output=True)

        waveform, sample_rate = torchaudio.load(io.BytesIO(audio), format='wav')
        mel_spec = MelSpectrogram(sample_rate=sample_rate, n_fft=2048, hop_length=512, n_mels=128)(waveform)

        epsilon = 1e-10
        log_mel_spec = torch.log2(mel_spec + epsilon)

        log_mel_spec = log_mel_spec[:, :, :128]

        spec_filename = output_path.rsplit('.', 1)[0] + '.png'
        save_spectogram(log_mel_spec[0], spec_filename, spec_max, spec_min)

    except ffmpeg.Error as e:
        print(f"Error processing {input_path}: {e}")


def process_video(class_name, video_file, spec_max, spec_min):
    """Process a video file to be split into 3s segments, be 16fps, 398x224 and have an audio sample rate of 20500Hz."""
    video_path = os.path.join("VAS", class_name, "videos", video_file)
    output_dir = os.path.join("data", class_name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        frame_rate, duration = get_video_info(video_path)
        segments = int(duration // 3)

        for i in range(segments):
            output_filename = f"{video_file.rsplit('.', 1)[0]}_{i}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            start_time = i * 3
            end_time = start_time + 3

            if frame_rate > 16:
                adjust_video_properties(video_path, output_path, start_time, end_time, spec_max, spec_min, 16, 22050, 224, 224)
                print(f"Processed {video_path}")
            else:
                print(f"Skipped {video_path} segment {i} due to frame rate <= 16 fps")

    except Exception as e:
        print(f"Error processing {video_path}: {e}")

def remove_corrupt_videos(root_dir):
    """Remove corrupt videos from the dataset."""
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        for video_file in os.listdir(class_dir):

            if video_file.endswith('.png'):
                continue

            video_path = os.path.join(class_dir, video_file)
            try:
                probe = ffmpeg.probe(video_path)
                video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']

                if not video_streams:
                    raise Exception("No video stream found")
                frame_count = int(video_streams[0].get('nb_frames', 0))

                if frame_count < 48:
                    raise Exception(f"Video has less than 48 frames: {frame_count}")
            except (ffmpeg.Error, Exception) as e:
                print(f"Removing corrupt video (and corresponding spectogram): {video_path} due to error: {e}")
                os.remove(video_path)
                os.remove(video_path.rsplit('.', 1)[0] + '.png')

def main():

    with open('spec_range.txt', 'r') as f:
        lines = f.readlines()
        spec_max = float(lines[0].strip())
        spec_min = float(lines[1].strip())

    classes = os.listdir("VAS")

    with ProcessPoolExecutor() as executor:
        for class_name in classes:
            video_dir = os.path.join("VAS", class_name, "videos")
            for video_file in os.listdir(video_dir):
                executor.submit(process_video, class_name, video_file, spec_max, spec_min)

    remove_corrupt_videos("data")


if __name__ == "__main__":
    main()
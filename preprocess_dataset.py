import os
import ffmpeg
from concurrent.futures import ProcessPoolExecutor
import av


def get_video_info(video_path):
    """Return the frame rate of the video using PyAV."""
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        frame_rate = stream.average_rate
        duration = container.duration / av.time_base
        return frame_rate, duration


def adjust_video_properties(input_path, output_path, start_time, end_time, target_fps=21.5, audio_sampling_rate=22050, target_width=398, target_height=224):
    """
    Adjusts the video frame rate to target_fps, converts audio to 22050 Hz mono,
    and resizes the video to fit the target height with padding to match the target width.
    """
    try:
        stream = ffmpeg.input(input_path, ss=start_time, to=end_time)
        video_stream = stream.video.filter('fps', fps=target_fps).filter('scale', '-2', target_height).filter('pad', target_width, target_height, '(ow-iw)/2', '(oh-ih)/2')
        audio_stream = stream.audio.filter('aformat', sample_rates=audio_sampling_rate, channel_layouts='mono')

        ffmpeg.output(video_stream, audio_stream, output_path, vcodec='libx264', acodec='aac').run(overwrite_output=True)

    except ffmpeg.Error as e:
        print(f"Error processing {input_path}: {e}")


def process_video(class_name, video_file):
    """Process a video file to be split into 3s segmens, be 21.5fps, 398x224 and have an audio sample rate of 20500Hz."""
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

            if frame_rate > 21.5:
                adjust_video_properties(video_path, output_path, start_time, end_time, 21.5, 22050, 398, 224)
                print(f"Processed {video_path}")
            else:
                print(f"Skipped {video_path} segment {i} due to frame rate <= 21.5 fps")

    except Exception as e:
        print(f"Error processing {video_path}: {e}")


def main():
    classes = ['baby', 'cough', 'dog', 'drum', 'fireworks', 'gun', 'hammer', 'sneeze']

    with ProcessPoolExecutor() as executor:
        for class_name in classes:
            video_dir = os.path.join("VAS", class_name, "videos")
            for video_file in os.listdir(video_dir):
                executor.submit(process_video, class_name, video_file)


if __name__ == "__main__":
    main()
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as running_mode
import os, subprocess
import time
import requests
from tqdm import trange
import moviepy.editor
from pathlib import Path
# from cap_from_youtube import cap_from_youtube


MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

frame_check_interval_original = 500  # 500/30 = 16.67 seconds
frame_check_interval_fine = 100  # 100/30 = 3.33 seconds
max_undetected_intervals = 3
minimum_clip_length = 1500  # roughly 50 seconds


def pull_clip(video_path: str, start_frame: int, end_frame: int):
    video = moviepy.editor.VideoFileClip(video_path)
    fps = video.fps
    clip = video.subclip(start_frame / fps, end_frame / fps)
    filename = str(start_frame) + ".mp4"
    clip.write_videofile(filename)
    return filename


def process_video_frame_by_frame(video_path: str, start_frame: int = 0):
    with vision.ObjectDetector.create_from_options(
            vision.ObjectDetectorOptions(
                base_options=python.BaseOptions(
#                    model_asset_path=os.path.join(os.path.dirname(__file__), "efficientdet_lite2_int8.tflite")),
                    model_asset_path=os.path.join(os.path.dirname(__file__), "ssd_mobilenet_v2.tflite")),
                running_mode=running_mode.IMAGE,
                category_allowlist=["cat", "dog"],
                score_threshold=0.2)) as detector:
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        consecutive_frames = 0
        grace_frames = 0
        frame_check_interval = frame_check_interval_original
        i = start_frame
        while i < total_frames:
            print("Processing frame " + str(i) + " of " + str(total_frames) + ", " + str(i / total_frames * 100) + "% complete")
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = detector.detect(mp_image)
            if detection_result.detections:
                consecutive_frames += frame_check_interval
                print(str(consecutive_frames) + " consecutive frames with confidence " + str(detection_result.detections[0].categories[0].score))
                grace_frames = 0
                if frame_check_interval == frame_check_interval_original:
                    frame_check_interval = frame_check_interval_fine
                    print("Slowing down to fine check interval")
            elif grace_frames < max_undetected_intervals and consecutive_frames > 0: # 9 * FRAME_CHECK_INTERVAL frames where no cat is detected, to allow for short interruptions / false negatives
                consecutive_frames += frame_check_interval
                grace_frames += 1
                print("Grace frame #" + str(grace_frames))
            else:
                if consecutive_frames >= minimum_clip_length:
                    first_clip_frame = i - consecutive_frames
                    last_clip_frame = i - (grace_frames * frame_check_interval)
                    print("Pulling clip from " + str(first_clip_frame) + " to " + str(last_clip_frame))
                    clip_filename = pull_clip(video_path, first_clip_frame, last_clip_frame)
                if frame_check_interval == frame_check_interval_fine:
                    frame_check_interval = frame_check_interval_original
                    print("Speeding up to original check interval")
                consecutive_frames = 0
                grace_frames = 0
            i += frame_check_interval

def mp4_merge():
    if subprocess.run(["which", "mp4_merge"]).returncode != 0:
        if subprocess.run(["uname", "-p"]).stdout == "aarch64":
            os.system("wget https://github.com/gyroflow/mp4-merge/releases/download/v0.1.8/mp4_merge-linux-arm64 -O /usr/bin/mp4_merge && chmod +x /usr/bin/mp4_merge")
        else:
            os.system("wget https://github.com/gyroflow/mp4-merge/releases/download/v0.1.8/mp4_merge-linux64 -O /usr/bin/mp4_merge && chmod +x /usr/bin/mp4_merge")
    command = "mp4_merge "
    files = []
    for file in os.listdir("."):
        if file.endswith(".mp4"):
            files.append(file)
    files.sort(key=lambda x: int(x.split(".")[0]))
    for file in files:
        command += file + " "
    os.system(command)

def pull_from_premiumize():
    result = subprocess.run(["rclone", "ls", "premiumize:climps/"], capture_output=True)
    files = result.stdout.decode().split("\n")
    files = [file for file in files if file != ""]  # remove empty strings
    file_names = []
    for file in files:
        file_name = file.split(" ")[-1]
        file_names.append(file_name)

    return file_names


def pull_from_google_drive():
    result = subprocess.run(["rclone", "ls", "google:Downloads/climps/"], capture_output=True)
    files = result.stdout.decode().split("\n")
    files = [file for file in files if file != ""]  # remove empty strings
    file_names = []
    for file in files:
        file_name = file.split(" ")[-1]
        file_names.append(file_name)

    return file_names

if __name__ == "__main__":
    os.chdir(str(Path.home()))
    # file_names = pull_from_premiumize()
    drive_file_names = pull_from_google_drive()
    # Because we may be running this on a VPS with little disk space, we will download one file at a time, process it,
    # and then delete it before moving on to the next file
    for file_name in drive_file_names:
        subprocess.run(["rclone", "copy", "google:Downloads/climps/" + file_name, str(Path.home()), "-P"])
        process_video_frame_by_frame(file_name)
        subprocess.run(["rm", str(Path.home()) + "/" + file_name])
        mp4_merge()
        for file in os.listdir("."):
            if "joined" in file:
                subprocess.run(["rclone", "copy", file, "google:Downloads/climps/", "-P"])

                subprocess.run(["rm", file])
                break
        subprocess.run(["rclone", "delete", "google:Downloads/climps/" + file_name, "-P"])
        subprocess.run(["rm", "-rf", "*.mp4"])

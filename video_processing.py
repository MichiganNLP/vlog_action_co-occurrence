#!/usr/bin/env python
import glob
import json
import os
import shutil
from collections import defaultdict
from typing import Sequence, Iterable
import subprocess
import cv2
import ffmpeg
import numpy as np
import torch
from rich.console import Console
from rich.progress import track

console = Console()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_video_by_frames(video_names: Iterable[str], new_video_names: Sequence[str]) -> None:
    count_skipped = 0
    for video, video_new in zip(video_names, track(new_video_names, description="Splitting videos into frames…")):
        print(f"Processing video {video}…")
        path_in = os.path.join("data/video_clips_sample", video + ".mp4")
        path_out = os.path.join("data/frames_sample", video_new)
        if not os.path.exists(path_in) or os.path.exists(path_out):
            count_skipped += 1
            print("Skipping splitting video by frames:", video_new)
            continue
        try:
            probe = ffmpeg.probe(path_in)
            time = float(probe['streams'][0]['duration'])
            width = probe['streams'][0]['width']
        except:
            print("Skipping corrupted video:", path_in)
            continue

        if not os.path.exists(path_out):
            os.makedirs(path_out)

        # split in middle frame
        # (
        #     ffmpeg
        #         .input(path_in, ss=time)
        #         .filter('scale', width, -1)
        #         .output(path_out + video_new + "_" + str(0) + '.jpeg', vframes=1)
        #         .run()
        # )
        # Set how many spots you want to extract a video from.
        parts = 4
        intervals = time // parts
        interval_list = [(i * intervals, (i + 1) * intervals) for i in range(parts)]
        for i, item in enumerate(interval_list):
            (
                ffmpeg
                    .input(path_in, ss=item[1])
                    .filter('scale', width, -1)
                    .output(path_out + video_new + "_" + str(i) + '.jpeg', vframes=1)
                    .run()
            )

    console.print(f"Skipped {count_skipped} clips from frame splitting", style="magenta")


def split_videos_into_frames(path: str) -> None:
    with open(path) as file:
        dict_action_clips_sample = json.load(file)
    video_names = []
    new_video_names = []
    for action, dict_video_times in dict_action_clips_sample.items():
        for dict_video_time in dict_video_times:
            video_name, time_s, time_e = dict_video_time.values()
            video_name = "+".join([video_name, time_s, time_e])
            new_video_name = "+".join(["_".join(action.split()), video_name])
            video_names.append(video_name)
            new_video_names.append(new_video_name)

    split_video_by_frames(video_names, new_video_names)


def filter_videos_by_motion(videos_path: str, problematic_videos_path: str, param_corr2d_coeff: float) -> None:
    list_videos = sorted(glob.glob(videos_path + "*.mp4"), key=os.path.getmtime)
    os.makedirs(problematic_videos_path, exist_ok=True)

    count = 0
    for video in track(list_videos, description="Filtering videos by motion…"):
        vidcap = cv2.VideoCapture(video)
        if not vidcap.isOpened():
            continue
        corr_list = []
        video_name = video.split("/")[-1]
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_nb_1 in range(0, length - 100, 100):
            vidcap.set(1, frame_nb_1)
            success, image = vidcap.read()
            if not success:
                continue
            gray_image_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            frame_nb_2 = frame_nb_1 + 100
            vidcap.set(1, frame_nb_2)
            success, image = vidcap.read()
            if not success:
                continue
            gray_image_2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            corr2_matrix = np.corrcoef(gray_image_1.reshape(-1), gray_image_2.reshape(-1))
            corr2 = corr2_matrix[0][1]
            corr_list.append(corr2)

        if np.median(corr_list) >= param_corr2d_coeff:
            count += 1
            shutil.move(video, problematic_videos_path + video_name)

    console.print(f"Filtered out {count} videos to {problematic_videos_path}", style="magenta")


def get_all_clips_for_action(input_file: str, output_file: str) -> None:
    with open(input_file) as json_file:
        dict_video_action_pairs_filtered = json.load(json_file)

    dict_action_clips = defaultdict(list)
    for video in dict_video_action_pairs_filtered:
        for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2) in \
                dict_video_action_pairs_filtered[video]:
            time_s, time_e = clip_a1
            time_s, time_e = time_s.split(".")[0], time_e.split(".")[0]
            if {"video": video, "time_s": time_s, "time_e": time_e} not in dict_action_clips[action_1]:
                dict_action_clips[action_1].append({"video": video, "time_s": time_s, "time_e": time_e})

            time_s, time_e = clip_a2
            time_s, time_e = time_s.split(".")[0], time_e.split(".")[0]
            if {"video": video, "time_s": time_s, "time_e": time_e} not in dict_action_clips[action_2]:
                dict_action_clips[action_2].append({"video": video, "time_s": time_s, "time_e": time_e})

    with open(output_file, 'w+') as file:
        json.dump(dict_action_clips, file)


def stats_videos() -> None:
    with open('data/coref_all_sentence_transcripts.json') as json_file:
        all_sentence_transcripts_rachel = json.load(json_file)
    nb_videos = len(all_sentence_transcripts_rachel)
    console.print(f"#Unique videos: {nb_videos}", style="magenta")

    with open('data/dict_action_clips.json') as json_file:
        dict_action_clips = json.load(json_file)

    all_clips = {"+".join([c["video"], c["time_s"], c["time_e"]])
                 for clips in dict_action_clips.values()
                 for c in clips}
    console.print(f"#Unique clips: {len(all_clips)}", style="magenta")

    all_action_clips = {"+".join([action, c["video"], c["time_s"], c["time_e"]])
                        for action, clips in dict_action_clips.items()
                        for c in clips}
    console.print(f"#Unique (action, clips): {len(all_action_clips)}", style="magenta")

    with open('data/dict_action_clips_sample.json') as json_file:
        dict_action_clips_sample = json.load(json_file)
    all_action_clips_sampled = {"+".join([action, c["video"], c["time_s"], c["time_e"]])
                                for action, clips in dict_action_clips_sample.items()
                                for c in clips}
    all_video_clips_sampled = {"+".join([c["video"], c["time_s"], c["time_e"]])
                          for _, clips in dict_action_clips_sample.items()
                          for c in clips}
    console.print(f"#Unique (action, clips) sampled: {len(all_action_clips_sampled)}", style="magenta")
    console.print(f"#Unique actions sampled: {len(dict_action_clips_sample.keys())}", style="magenta")
    console.print(f"#Unique videos sampled: {len(all_video_clips_sampled)}", style="magenta")

    folder = 'data/video_clips_sample'
    sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    set_actions_downloaded = set()
    for video_name in sub_folders:
        action = video_name.split("+")[0].replace("_", " ")
        set_actions_downloaded.add(action)
    console.print(f"#Unique (action, clips) downloaded: {len(sub_folders)}", style="magenta")
    console.print(f"#Unique actions downloaded: {len(set_actions_downloaded)}", style="magenta")


def get_video_diff() -> None:
    with open('data/dict_action_clips_sample.json') as json_file:
        dict_action_clips = json.load(json_file)

    all_videos = {"+".join([dict_["video"], dict_["time_s"], dict_["time_e"]]) + ".mp4"
                  for values in dict_action_clips.values()
                  for dict_ in values}
    print(len(all_videos))
    all_videos_downloaded = {video_name.replace('data/video_clips_sample/', '') for video_name in
                             glob.glob("data/video_clips_sample/*.mp4") + glob.glob("data/filtered_videos/*.mp4")}
    print(len(all_videos_downloaded))
    not_downloaded = all_videos - all_videos_downloaded
    not_in_dict = all_videos_downloaded - all_videos
    print(len(not_downloaded))
    print(len(not_in_dict))
    # for video_name in not_in_dict:
    #     shutil.move('data/video_clips_sample/' + video_name, 'data/videos_old/' + video_name)

    dict_action_clips_sample_remained = defaultdict(list)
    for action, values in dict_action_clips.items():
        for dict_ in values:
            if "+".join([dict_["video"], dict_["time_s"], dict_["time_e"]]) + ".mp4" in not_downloaded:
                dict_action_clips_sample_remained[action].append(dict_)
    with open('data/dict_action_clips_sample_remained.json', 'w+') as fp:
        json.dump(dict_action_clips_sample_remained, fp)


def sample_videos(input_file: str, output_file: str, max_videos_per_action: int) -> None:
    with open(input_file) as json_file:
        dict_action_clips = json.load(json_file)

    dict_action_clips_sample = {action: dict_action_clips[action][:max_videos_per_action]
                                for action in dict_action_clips.keys()}

    with open(output_file, 'w+') as fp:
        json.dump(dict_action_clips_sample, fp)
    print(len(dict_action_clips_sample))


def main() -> None:
    pass
    # get_video_diff()

    # get_all_clips_for_action(input_file="data/dict_video_action_pairs_filtered_by_link.json",
    #                          output_file="data/dict_action_clips.json")
    # sample_videos(input_file='data/dict_action_clips.json', output_file='data/dict_action_clips_sample.json',
    #               max_videos_per_action=10)
    # stats_videos()

    # TODO: Add 5 seconds before and after time to account for misalignment
    # subprocess.run(["./download_video_clips.sh", "data/dict_action_clips_sample.json", "data/video_clips_sample"])
    # subprocess.run(["./download_videos.sh", "data/dict_action_clips_sample.json", "data/videos_sample"])

    # filter_videos_by_motion(path_videos="data/video_clips_sample/", path_problematic_videos="data/filtered_videos/",
    #                         PARAM_CORR2D_COEFF=0.9)

    # split_videos_into_frames(input_file="data/dict_action_clips_sample.json")




if __name__ == '__main__':
    main()

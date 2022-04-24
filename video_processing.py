#!/usr/bin/env python
import glob
import json
import os
import shutil
from collections import defaultdict
from typing import Mapping, Any, Sequence, Callable, Tuple, Iterable

import clip
import cv2
import ffmpeg
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rich.console import Console
from rich.progress import track
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader

console = Console()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WORKERS = len(os.sched_getaffinity(0)) // max(torch.cuda.device_count(), 1)

FRAME_COUNT = 4

TEMPLATES = [
    "a photo of a person {}.",
    "a video of a person {}.",
    "a example of a person {}.",
    "a demonstration of a person {}.",
    "a photo of the person {}.",
    "a video of the person {}.",
    "a example of the person {}.",
    "a demonstration of the person {}.",
    "a photo of a person using {}.",
    "a video of a person using {}.",
    "a example of a person using {}.",
    "a demonstration of a person using {}.",
    "a photo of the person using {}.",
    "a video of the person using {}.",
    "a example of the person using {}.",
    "a demonstration of the person using {}.",
    "a photo of a person doing {}.",
    "a video of a person doing {}.",
    "a example of a person doing {}.",
    "a demonstration of a person doing {}.",
    "a photo of the person doing {}.",
    "a video of the person doing {}.",
    "a example of the person doing {}.",
    "a demonstration of the person doing {}.",
    "a photo of a person during {}.",
    "a video of a person during {}.",
    "a example of a person during {}.",
    "a demonstration of a person during {}.",
    "a photo of the person during {}.",
    "a video of the person during {}.",
    "a example of the person during {}.",
    "a demonstration of the person during {}.",
    "a photo of a person performing {}.",
    "a video of a person performing {}.",
    "a example of a person performing {}.",
    "a demonstration of a person performing {}.",
    "a photo of the person performing {}.",
    "a video of the person performing {}.",
    "a example of the person performing {}.",
    "a demonstration of the person performing {}.",
    "a photo of a person practicing {}.",
    "a video of a person practicing {}.",
    "a example of a person practicing {}.",
    "a demonstration of a person practicing {}.",
    "a photo of the person practicing {}.",
    "a video of the person practicing {}.",
    "a example of the person practicing {}.",
    "a demonstration of the person practicing {}.",
]


def split_video_by_frames(video_names: Iterable[str], new_video_names: Sequence[str]) -> None:
    count_skipped = 0
    for video, video_new in zip(video_names, track(new_video_names, description="Splitting videos into frames…")):
        print(f"Processing video {video}…")
        path_in = os.path.join("data/videos_sample", video + ".mp4")
        path_out = os.path.join("data/videos_sample2", video_new)
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
        # parts = 4
        # intervals = time // parts
        # interval_list = [(i * intervals, (i + 1) * intervals) for i in range(parts)]
        # for i, item in enumerate(interval_list):
        #     (
        #         ffmpeg
        #             .input(path_in, ss=item[1])
        #             .filter('scale', width, -1)
        #             .output(path_out + video_new + "_" + str(i) + '.jpeg', vframes=1)
        #             .run()
        #     )
        (ffmpeg
         .input(path_in)
         .filter('fps', fps='1')
         .output(path_out + video_new + "_%d" + '.jpeg',
                 start_number=0)
         .overwrite_output()
         .run(quiet=True)
         )

    # print(interval_list, intervals, time)
    console.print(f"Skipped {count_skipped} clips from frame splitting", style="magenta")


def split_videos_into_frames(path: str) -> None:
    with open(path) as file:
        dict_test_clip = json.load(file)
    video_names = []
    new_video_names = []
    for action, dict_video_times in dict_test_clip.items():
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

        # print(video_name, np.median(corr_list))
        if np.median(corr_list) >= param_corr2d_coeff:
            count += 1
            shutil.move(video, problematic_videos_path + video_name)

    console.print(f"Filtered out {count} videos to {problematic_videos_path}", style="magenta")


def get_all_clips_for_action(output_path: str) -> None:
    with open('data/dict_video_action_pairs_filtered_by_link.json') as json_file:
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

    with open(output_path, 'w+') as file:
        json.dump(dict_action_clips, file)


def save_clip_features(video_features: torch.Tensor, text_features: torch.Tensor, video_names: Sequence[str],
                       output_path: str = "data/clip_features3.pt") -> None:
    torch.save({video_name: {"visual": video_feature, "text": text_feature}
                for video_feature, text_feature, video_name in zip(video_features, text_features, video_names)},
               output_path)


class VideoDataset(Dataset):
    def __init__(self, directories: Sequence[str], transform: Callable[[Image.Image], torch.Tensor],
                 tokenizer: Callable[[str], torch.Tensor]) -> None:
        super().__init__()
        self.directories = directories
        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i: int) -> Mapping[str, Any]:
        directory = self.directories[i]

        frame_filenames = sorted([filename
                                  for filename in os.listdir(directory)
                                  if filename.lower().endswith((".jpg", ".jpeg", ".png"))])
        assert len(frame_filenames) == FRAME_COUNT

        name = os.path.splitext(frame_filenames[0])[0]
        action = " ".join(name.split("+", maxsplit=1)[0].split("_"))

        return {
            "text_tokens": self.tokenizer(template.format(action) for template in TEMPLATES),  # noqa
            "video": torch.stack([self.transform(Image.open(os.path.join(directory, frame_filename)))
                                  for frame_filename in frame_filenames]),
        }

    def __len__(self) -> int:
        return len(self.directories)


def run_clip(path: str, data_dir: str = "data/videos_sample") -> Tuple[torch.Tensor, torch.Tensor, Sequence[str]]:
    with open(path) as file:
        dict_test_clip = json.load(file)

    video_names = []
    directories = []
    for action, dict_video_times in dict_test_clip.items():
        for dict_video_time in dict_video_times:
            video, time_s, time_e = dict_video_time.values()
            video_name = "+".join(["_".join(action.split()), video, time_s, time_e])
            video_path = os.path.join(data_dir, video_name)
            if os.path.exists(video_path):  # Sometimes there are missing videos from YouTube, so we check.
                video_names.append(video_name)
                directories.append(video_path)

    model, transform = clip.load("ViT-L/14", device=DEVICE, jit=True)

    dataset = VideoDataset(directories, transform=transform, tokenizer=clip.tokenize)
    data_loader = DataLoader(dataset, batch_size=96, num_workers=NUM_WORKERS, pin_memory=True)

    video_feature_list = []
    text_feature_list = []

    with torch.inference_mode():
        for batch in track(data_loader, description="Extracting data for CLIP…"):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            video = batch["video"]
            video = video.view(-1, *video.shape[2:])
            video_features = model.encode_image(video)

            token_ids = batch["text_tokens"]
            token_ids = token_ids.view(-1, *token_ids.shape[2:])
            text_features = model.encode_text(token_ids)

            # Get the mean of frame features for each video.
            reshaped_video_features = video_features.reshape(video_features.shape[0] // FRAME_COUNT, FRAME_COUNT, -1)
            video_features = reshaped_video_features.mean(dim=1)

            # Get the mean of text features for the different prompts, for each video.
            reshaped_text_features = text_features.reshape(text_features.shape[0] // len(TEMPLATES), len(TEMPLATES), -1)
            text_features = reshaped_text_features.mean(dim=1)

            assert video_features.shape == text_features.shape

            video_feature_list.append(video_features)
            text_feature_list.append(text_features)

    video_features = torch.cat(video_feature_list)
    text_features = torch.cat(text_feature_list)

    # video_features /= video_features.norm(dim=-1, keepdim=True)
    # text_features /= text_features.norm(dim=-1, keepdim=True)

    # similarity = video_features @ text_features.T
    # print(similarity)

    return video_features, text_features, video_names


def stats_videos() -> None:
    with open('data/coref_all_sentence_transcripts.json') as json_file:
        all_sentence_transcripts_rachel = json.load(json_file)
    nb_videos = len(all_sentence_transcripts_rachel)
    console.print(f"#Unique videos: {nb_videos}", style="magenta")

    with open('data/dict_action_clips.json') as json_file:
        # with open('data/dict_action_clips_sample.json') as json_file:
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
    all_videos_sampled = {"+".join([c["video"], c["time_s"], c["time_e"]])
                          for _, clips in dict_action_clips_sample.items()
                          for c in clips}
    console.print(f"#Unique (action, clips) sampled: {len(all_action_clips_sampled)}", style="magenta")
    console.print(f"#Unique actions sampled: {len(dict_action_clips_sample.keys())}", style="magenta")
    console.print(f"#Unique videos sampled: {len(all_videos_sampled)}", style="magenta")


def get_video_diff() -> None:
    with open('data/dict_action_clips_sample.json') as json_file:
        dict_action_clips = json.load(json_file)

    all_videos = {"+".join([dict_["video"], dict_["time_s"], dict_["time_e"]]) + ".mp4"
                  for values in dict_action_clips.values()
                  for dict_ in values}
    print(len(all_videos))
    all_videos_downloaded = {video_name.replace('data/videos_sample/', '') for video_name in
                             glob.glob("data/videos_sample/*.mp4") + glob.glob("data/filtered_videos/*.mp4")}
    # all_videos_downloaded = {video_name.replace('data/videos_sample/', '')
    #                          for video_name in glob.glob("data/videos_sample/*.mp4")}
    print(len(all_videos_downloaded))
    not_downloaded = all_videos - all_videos_downloaded
    print(len(not_downloaded))

    dict_action_clips_sample_remained = defaultdict(list)
    for action, values in dict_action_clips.items():
        for dict_ in values:
            if "+".join([dict_["video"], dict_["time_s"], dict_["time_e"]]) + ".mp4" in not_downloaded:
                dict_action_clips_sample_remained[action].append(dict_)
    with open('data/dict_action_clips_sample_remained.json', 'w+') as fp:
        json.dump(dict_action_clips_sample_remained, fp)


def sample_videos() -> None:
    with open('data/dict_action_clips.json') as json_file:
        dict_action_clips = json.load(json_file)
    nb_videos = 10
    dict_action_clips_sample = {action: dict_action_clips[action][:nb_videos] for action in dict_action_clips.keys()}
    dict_action_clips_sample_test = {action: dict_action_clips[action][:nb_videos] for action in
                                     dict_action_clips.keys()
                                     if action in ["clean sink", 'clean kitchen', 'put music']}
    with open('data/dict_action_clips_sample.json', 'w+') as fp:
        json.dump(dict_action_clips_sample, fp)

    with open('data/dict_action_clips_sample_test.json', 'w+') as fp:
        json.dump(dict_action_clips_sample_test, fp)
    # 730 * 10 .. 7297 - put rose petal has 7 videos


def test_clip() -> None:
    nodes = pd.read_csv('data/graph/all_stsbrt_nodes.csv', index_col=0)
    # print(nodes.index)
    action1 = nodes.loc[['clean sink']].to_numpy()
    action2 = nodes.loc[['scrub sink']].to_numpy()
    action3 = nodes.loc[['clean kitchen']].to_numpy()
    action4 = nodes.loc[['compete with dollar store']].to_numpy()
    action5 = nodes.loc[['put music']].to_numpy()
    action6 = nodes.loc[['add beet']].to_numpy()
    print(cosine_similarity(action1, action2), cosine_similarity(action1, action3), cosine_similarity(action1, action4),
          cosine_similarity(action1, action5), cosine_similarity(action1, action6))

    # nodes = pd.read_csv('data/graph/all_txtclip_nodes.csv', index_col=0)
    # action1 = nodes.loc[['clean sink']].to_numpy()
    # action2 = nodes.loc[['clean kitchen']].to_numpy()
    # action3 = nodes.loc[['compete with dollar store']].to_numpy()
    # action4 = nodes.loc[['put music']].to_numpy()
    # print(cosine_similarity(action1, action2), cosine_similarity(action1, action3),
    #       cosine_similarity(action1, action4))

    nodes = pd.read_csv('data/graph/all_visclip_nodes.csv', index_col=0)
    action1 = nodes.loc[['clean sink']].to_numpy()
    action2 = nodes.loc[['scrub sink']].to_numpy()
    action3 = nodes.loc[['clean kitchen']].to_numpy()
    action4 = nodes.loc[['compete with dollar store']].to_numpy()
    action5 = nodes.loc[['put music']].to_numpy()
    action6 = nodes.loc[['add beet']].to_numpy()
    print(cosine_similarity(action1, action2), cosine_similarity(action1, action3), cosine_similarity(action1, action4),
          cosine_similarity(action1, action5), cosine_similarity(action1, action6))

    nodes = pd.read_csv('data/graph/all_weighted_visclip_nodes.csv', index_col=0)
    action1 = nodes.loc[['clean sink']].to_numpy()
    action2 = nodes.loc[['scrub sink']].to_numpy()
    action3 = nodes.loc[['clean kitchen']].to_numpy()
    action4 = nodes.loc[['compete with dollar store']].to_numpy()
    action5 = nodes.loc[['put music']].to_numpy()
    action6 = nodes.loc[['add beet']].to_numpy()
    print(cosine_similarity(action1, action2), cosine_similarity(action1, action3), cosine_similarity(action1, action4),
          cosine_similarity(action1, action5), cosine_similarity(action1, action6))

    # nodes = pd.read_csv('data/graph/all_avgclip_nodes.csv', index_col=0)
    # action1 = nodes.loc[['clean sink']].to_numpy()
    # action2 = nodes.loc[['clean kitchen']].to_numpy()
    # action3 = nodes.loc[['compete with dollar store']].to_numpy()
    # action4 = nodes.loc[['put music']].to_numpy()
    # print(cosine_similarity(action1, action2), cosine_similarity(action1, action3),
    #       cosine_similarity(action1, action4))


def main() -> None:
    # get_video_diff()

    # get_all_clips_for_action(output_file="data/dict_action_clips.json") #dict_action_clips_sample
    # sample_videos() # max 10 videos/ action
    # stats_videos()

    # TODO: Add 5 seconds before and after time to account for misalignment
    # subprocess.run(["./download_videos.sh",
    #                 "data/dict_action_clips_sample.json", "data/videos_sample"])
    # "data/dict_action_clips_sample.json", "data/url_list_sample.txt", "data/videos_sample"])
    # filter_videos_by_motion(path_videos="data/videos_sample/", path_problematic_videos="data/filtered_videos/",
    #                         PARAM_CORR2D_COEFF=0.9)
    # split_videos_into_frames(input_file="data/dict_action_clips_sample.json") # dict_action_clips_sample_remained

    video_features, text_features, video_names = run_clip(path="data/dict_action_clips_sample.json")
    assert len(video_features) == len(text_features) == len(video_names)
    save_clip_features(video_features, text_features, video_names)

    # test_clip()


if __name__ == '__main__':
    main()

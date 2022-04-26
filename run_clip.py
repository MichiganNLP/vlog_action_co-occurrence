#!/usr/bin/env python
from __future__ import annotations

import datetime
import json
import logging
import os
import time
from typing import Mapping, Any, Sequence, Callable

import clip
import decord
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rich.console import Console
from rich.progress import track
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

LOGGER = logging.getLogger(__name__)

console = Console()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEMPERATURE = 100

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


def save_clip_features(feature_dicts: Sequence[Mapping[str, Any]], output_path: str) -> None:
    torch.save(feature_dicts, output_path)


decord.bridge.set_bridge("torch")


def time_to_indices(time: float | Sequence[float], video_reader: decord.VideoReader) -> np.ndarray:
    times = video_reader.get_frame_timestamp(range(len(video_reader))).mean(-1)
    indices = np.searchsorted(times, time)
    return np.where(np.bitwise_or(indices == 0, times[indices] - time <= time - times[indices - 1]), indices,
                    indices - 1)


def time_string_to_seconds(time_str: str) -> float:
    t = time.strptime(time_str.split(",", maxsplit=1)[0], "%H:%M:%S")
    return datetime.timedelta(hours=t.tm_hour, minutes=t.tm_min, seconds=t.tm_sec).total_seconds()


class VideoDataset(Dataset):
    def __init__(self, metadata_path: str, videos_dir: str,
                 transform: Callable[[Image.Image], torch.Tensor], tokenizer: Callable[[str], torch.Tensor],
                 num_frames: int = 4, extra_time: int = 0) -> None:
        super().__init__()

        with open(metadata_path) as file:
            action_clips_dict = json.load(file)

        self.action_clips_flatten = []
        for action, video_time_dicts in track(action_clips_dict.items(), total=len(action_clips_dict),
                                              description="Checking the video files"):
            for video_time_dict in video_time_dicts:
                parent_video_id, start_time, end_time = video_time_dict.values()
                path = os.path.join(videos_dir, parent_video_id + ".mp4")
                if os.path.exists(path):
                    try:
                        decord.VideoReader(path, num_threads=1)

                        self.action_clips_flatten.append({
                            "action": action,
                            "parent_video_id": parent_video_id,
                            "start_time": time_string_to_seconds(start_time),
                            "end_time": time_string_to_seconds(end_time),
                            "path": path,
                        })
                    except decord.DECORDError:
                        LOGGER.warning(f"Can't open video {path}.")
                else:
                    LOGGER.warning(f"Missing video {path}.")

        self.transform = transform
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.extra_time = extra_time

    def __getitem__(self, i: int) -> Mapping[str, Any]:
        action_clip = self.action_clips_flatten[i]

        action = action_clip["action"]
        parent_video_id = action_clip["parent_video_id"]
        start_time = action_clip["start_time"]
        end_time = action_clip["end_time"]
        path = action_clip["path"]

        try:
            video_reader = decord.VideoReader(path, num_threads=1)

            start_frame_idx, end_frame_idx = time_to_indices([start_time - self.extra_time, end_time + self.extra_time],
                                                             video_reader)
            start_frame_idx = max(start_frame_idx, 0)
            end_frame_idx = min(end_frame_idx, len(video_reader) - 1)

            indices = torch.linspace(start_frame_idx, end_frame_idx, steps=self.num_frames).round().to(torch.int)
            video = video_reader.get_batch(indices)
        except decord.DECORDError:
            LOGGER.error(f"An error occurred when trying to read the video with path {path}.")
            video = torch.zeros(self.num_frames, 256, 256, 3)

        # To save a frame to later visualize it:
        # plt.imsave("abc.png", video[0].numpy())

        return {
            "action": action,
            "parent_video_id": parent_video_id,
            "start_time": start_time,
            "end_time": end_time,
            "path": path,
            "text_tokens": self.tokenizer(template.format(action) for template in TEMPLATES),  # noqa
            "video": self.transform(video),
        }

    def __len__(self) -> int:
        return len(self.action_clips_flatten)


class ConvertBHWCtoBCHW(nn.Module):
    def forward(self, v: torch.Tensor) -> torch.Tensor:  # noqa
        return v.permute(0, 3, 1, 2)


def extract_clip_features(metadata_path: str, videos_dir: str) -> Sequence[Mapping[str, Any]]:
    model = clip.load("ViT-L/14", device=DEVICE, jit=True)[0]

    input_size = model.input_resolution.item()

    dtype = next((p.dtype for p in model.parameters() if p.dtype is torch.float16), torch.float)

    transform = T.Compose([
        ConvertBHWCtoBCHW(),
        T.ConvertImageDtype(dtype),
        T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(input_size),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    dataset = VideoDataset(metadata_path=metadata_path, videos_dir=videos_dir, transform=transform,
                           tokenizer=clip.tokenize)
    data_loader = DataLoader(dataset, batch_size=96, num_workers=NUM_WORKERS, pin_memory=True)

    feature_dicts = []

    with torch.inference_mode():
        for batch in track(data_loader, description="Extracting CLIP features"):
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

            for action, start_time, end_time, path, parent_video_id, visual, text \
                in zip(batch["action"], batch["start_time"], batch["end_time"], batch["path"], batch["parent_video_id"],
                       video_features, text_features):
                feature_dicts.append({
                    "action": action,
                    "start_time": start_time,
                    "end_time": end_time,
                    "path": path,
                    "parent_video_id": parent_video_id,
                    "visual_features": visual,
                    "text_features": text,
                })

    return feature_dicts


def test_clip() -> None:
    nodes = pd.read_csv('data/graph/all_stsbrt_nodes.csv', index_col=0)
    # print(nodes.index)
    action1 = nodes.loc[['clean sink']].to_numpy()
    action2 = nodes.loc[['wash sink']].to_numpy()
    action3 = nodes.loc[['clean kitchen']].to_numpy()
    action4 = nodes.loc[['combine gallon bag']].to_numpy()
    action5 = nodes.loc[['put music']].to_numpy()
    action6 = nodes.loc[['add barbecue']].to_numpy()
    print(cosine_similarity(action1, action2), cosine_similarity(action1, action3), cosine_similarity(action1, action4),
          cosine_similarity(action1, action5), cosine_similarity(action1, action6))

    nodes = pd.read_csv('data/graph/all_txtclip_nodes.csv', index_col=0)
    action1 = nodes.loc[['clean sink']].to_numpy()
    action2 = nodes.loc[['wash sink']].to_numpy()
    action3 = nodes.loc[['clean kitchen']].to_numpy()
    action4 = nodes.loc[['combine gallon bag']].to_numpy()
    action5 = nodes.loc[['put music']].to_numpy()
    action6 = nodes.loc[['add barbecue']].to_numpy()
    print(cosine_similarity(action1, action2), cosine_similarity(action1, action3), cosine_similarity(action1, action4),
          cosine_similarity(action1, action5), cosine_similarity(action1, action6))

    nodes = pd.read_csv('data/graph/all_visclip_nodes.csv', index_col=0)
    action1 = nodes.loc[['clean sink']].to_numpy()
    action2 = nodes.loc[['wash sink']].to_numpy()
    action3 = nodes.loc[['clean kitchen']].to_numpy()
    action4 = nodes.loc[['combine gallon bag']].to_numpy()
    action5 = nodes.loc[['put music']].to_numpy()
    action6 = nodes.loc[['add barbecue']].to_numpy()
    print(cosine_similarity(action1, action2), cosine_similarity(action1, action3), cosine_similarity(action1, action4),
          cosine_similarity(action1, action5), cosine_similarity(action1, action6))

    nodes = pd.read_csv('data/graph/all_weighted_visclip_nodes.csv', index_col=0)
    action1 = nodes.loc[['clean sink']].to_numpy()
    action2 = nodes.loc[['wash sink']].to_numpy()
    action3 = nodes.loc[['clean kitchen']].to_numpy()
    action4 = nodes.loc[['combine gallon bag']].to_numpy()
    action5 = nodes.loc[['put music']].to_numpy()
    action6 = nodes.loc[['add barbecue']].to_numpy()
    print(cosine_similarity(action1, action2), cosine_similarity(action1, action3), cosine_similarity(action1, action4),
          cosine_similarity(action1, action5), cosine_similarity(action1, action6))


    nodes = pd.read_csv('data/graph/all_avgclip_nodes.csv', index_col=0)
    action1 = nodes.loc[['clean sink']].to_numpy()
    action2 = nodes.loc[['wash sink']].to_numpy()
    action3 = nodes.loc[['clean kitchen']].to_numpy()
    action4 = nodes.loc[['combine gallon bag']].to_numpy()
    action5 = nodes.loc[['put music']].to_numpy()
    action6 = nodes.loc[['add barbecue']].to_numpy()
    print(cosine_similarity(action1, action2), cosine_similarity(action1, action3), cosine_similarity(action1, action4),
          cosine_similarity(action1, action5), cosine_similarity(action1, action6))


def evaluate_clip_embeddings(feature_dicts: Sequence[Mapping[str, Any]]) -> None:
    with torch.inference_mode():
        video_features = torch.stack([f["visual_features"] for f in feature_dicts]).to(DEVICE)
        text_features = torch.stack([f["text_features"] for f in feature_dicts]).to(DEVICE)

        video_features /= video_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = video_features @ text_features.T

        target = torch.arange(len(similarity), device="cuda")

        sorted_predicted_positions = similarity.argsort(dim=-1, descending=True)
        ranks = (sorted_predicted_positions == target.unsqueeze(dim=-1)).nonzero(as_tuple=True)[1]  # noqa

        print("Mean rank:", ranks.to(torch.float).mean())
        print("Median rank:", ranks.median())

        print("R@1:", (similarity.argmax(dim=-1) == target).to(torch.float).mean())
        print("R@5:", (similarity.topk(5)[1] == target.unsqueeze(dim=-1)).sum(dim=-1).to(torch.float).mean())  # noqa
        print("R@10:", (similarity.topk(10)[1] == target.unsqueeze(dim=-1)).sum(dim=-1).to(torch.float).mean())  # noqa

        probs = (similarity * TEMPERATURE).softmax(dim=-1)

        ground_truth_probs = probs.diag()

        print("Mean ground truth probability:", ground_truth_probs.mean())
        print("Median ground truth probability:", ground_truth_probs.median())


def stats_clip(path: str) -> None:
    features_dict = torch.load(path)
    actions = set()
    for video_name in features_dict:
        action_in_dict = video_name.split("+")[0].replace("_", " ")
        actions.add(action_in_dict)
    console.print(f"#Unique (action, clips) in CLIP features dict: {len(features_dict)}", style="magenta")
    console.print(f"#Unique actions in CLIP features dict: {len(actions)}", style="magenta")

    with open("data/dict_action_clips_sample.json") as file:
        dict_action_clips_sample = json.load(file)
    console.print(f"#Unique actions in dict_action_clips_sample: {len(dict_action_clips_sample)}", style="magenta")

    folder = 'data/video_clips_sample'
    sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    set_actions_downloaded = set()
    for video_name in sub_folders:
        action = video_name.split("+")[0].replace("_", " ")
        set_actions_downloaded.add(action)
    console.print(f"#Unique (action, clips) downloaded: {len(sub_folders)}", style="magenta")
    console.print(f"#Unique actions downloaded: {len(set_actions_downloaded)}", style="magenta")

    # for action in actions:
    #     if action not in dict_action_clips_sample:
    #         print(action)


def main() -> None:
    pass
    feature_dicts = extract_clip_features(metadata_path="data/dict_action_clips_sample.json",
                                          videos_dir="data/videos_test")
    output_path = "data/clip_features4.pt"
    save_clip_features(feature_dicts, output_path=output_path)

    stats_clip(path=output_path)
    evaluate_clip_embeddings(feature_dicts)
    test_clip()


if __name__ == '__main__':
    main()

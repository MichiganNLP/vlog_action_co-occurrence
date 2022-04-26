#!/usr/bin/env python
from __future__ import annotations

import json
import logging
import os
from typing import Mapping, Any, Sequence, Callable, Tuple

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


def save_clip_features(video_features: torch.Tensor, text_features: torch.Tensor, video_names: Sequence[str],
                       output_path: str) -> None:
    torch.save({video_name: {"visual": video_feature, "text": text_feature}
                for video_feature, text_feature, video_name in zip(video_features, text_features, video_names)},
               output_path)


decord.bridge.set_bridge("torch")


def time_to_indices(time: float | Sequence[float], video_reader: decord.VideoReader) -> np.ndarray:
    times = video_reader.get_frame_timestamp(range(len(video_reader))).mean(-1)
    indices = np.searchsorted(times, time)
    return np.where(np.bitwise_or(indices == 0, times[indices] - time <= time - times[indices - 1]), indices,
                    indices - 1)


class VideoDataset(Dataset):
    def __init__(self, video_paths: Sequence[str], actions: Sequence[str],
                 transform: Callable[[Image.Image], torch.Tensor], tokenizer: Callable[[str], torch.Tensor],
                 num_frames: int = 4) -> None:
        super().__init__()
        self.video_paths = video_paths
        self.actions = actions
        self.transform = transform
        self.tokenizer = tokenizer
        self.num_frames = num_frames

    def __getitem__(self, i: int) -> Mapping[str, Any]:
        video_path = self.video_paths[i]
        action = self.actions[i]

        try:
            video_reader = decord.VideoReader(video_path, num_threads=1)
            indices = torch.linspace(0, len(video_reader) - 1, steps=self.num_frames).round().to(torch.int)
            video = video_reader.get_batch(indices)
        except decord.DECORDError:
            LOGGER.error(f"An error occurred when trying to read the video with path {video_path}.")
            video = torch.zeros(self.num_frames, 256, 256, 3)

        # To save a frame to later visualize it:
        # plt.imsave("abc.png", video[0].numpy())

        return {
            "text_tokens": self.tokenizer(template.format(action) for template in TEMPLATES),  # noqa
            "video": self.transform(video),
        }

    def __len__(self) -> int:
        return len(self.video_paths)


class ConvertBHWCtoBCHW(nn.Module):
    def forward(self, v: torch.Tensor) -> torch.Tensor:  # noqa
        return v.permute(0, 3, 1, 2)


def extract_clip_features(path: str,
                          data_dir: str = "data/video_clips_sample") -> Tuple[torch.Tensor, torch.Tensor, Sequence[str]]:
    with open(path) as file:
        action_clips_dict = json.load(file)

    # video_ids = []
    video_names = []
    video_paths = []
    actions = []
    for action, video_time_dicts in track(action_clips_dict.items(), total=len(action_clips_dict),
                                          description="Checking the video files"):
        for video_time_dict in video_time_dicts:
            video_id = "+".join(video_time_dict.values())
            video_name = "+".join([action.replace(" ", "_")] + list(video_time_dict.values()))
            video_path = os.path.join(data_dir, video_id + ".mp4")
            if os.path.exists(video_path):  # Sometimes there are missing videos from YouTube, so we check.
                # video_ids.append(video_id)
                video_names.append(video_name)
                video_paths.append(video_path)
                actions.append(action)

    print(len(set(actions)), len(set(video_paths)))
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

    dataset = VideoDataset(video_paths, actions, transform=transform, tokenizer=clip.tokenize)
    data_loader = DataLoader(dataset, batch_size=96, num_workers=NUM_WORKERS, pin_memory=True)

    video_feature_list = []
    text_feature_list = []

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

            video_feature_list.append(video_features)
            text_feature_list.append(text_features)

    video_features = torch.cat(video_feature_list)
    text_features = torch.cat(text_feature_list)

    # video_features /= video_features.norm(dim=-1, keepdim=True)
    # text_features /= text_features.norm(dim=-1, keepdim=True)

    # similarity = video_features @ text_features.T
    # print(similarity)

    return video_features, text_features, video_names


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


def evaluate_clip_embeddings(path: str) -> None:
    features_dict = torch.load(path)

    with torch.inference_mode():
        video_features = torch.stack([b["visual"] for b in features_dict.values()]).to(DEVICE)
        text_features = torch.stack([b["text"] for b in features_dict.values()]).to(DEVICE)

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


def stats_clip(input_file):
    features_dict = torch.load(input_file)
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
    # video_features, text_features, video_names = extract_clip_features(path="data/dict_action_clips_sample.json")
    # assert len(video_features) == len(text_features) == len(video_names)
    # output_path = "data/clip_features4.pt"
    # save_clip_features(video_features, text_features, video_names, output_path=output_path)

    # stats_clip(input_file=output_path)
    # evaluate_clip_embeddings(output_path)
    # test_clip()


if __name__ == '__main__':
    main()

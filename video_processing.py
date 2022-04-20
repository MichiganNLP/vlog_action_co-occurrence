import glob
import json
import os
import shutil
from collections import defaultdict
from typing import Mapping, Any, Sequence, Callable

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
from torch.utils.data import Dataset

console = Console()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_video_by_frames(video_names, new_video_names):
    for video, video_new in track(list(zip(video_names, new_video_names)),
                                  description="Splitting videos into frames..."):
        print(f"Processing video {video} ...")
        path_in = "data/videos_sample/" + video + ".mp4"
        path_out = "data/videos_sample2/" + video_new + "/"
        count_skipped = 0
        if not os.path.exists(path_in) or os.path.exists(path_out):
            count_skipped += 1
            print(f"Skipping splitting video by frames: {video_new}")
            continue
        try:
            probe = ffmpeg.probe(path_in)
            time = float(probe['streams'][0]['duration'])
            width = probe['streams'][0]['width']
        except:
            print(f"Skipping corrupted video: {path_in}")
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


def split_videos_into_frames(input_file):
    with open(input_file) as json_file:
        dict_test_clip = json.load(json_file)
    video_names = []
    new_video_names = []
    for action in dict_test_clip:
        for dict_video_time in dict_test_clip[action]:
            video_name, time_s, time_e = dict_video_time.values()
            video_name = "+".join([video_name, time_s, time_e])
            new_video_name = "+".join(["_".join(action.split()), video_name])
            video_names.append(video_name)
            new_video_names.append(new_video_name)

    split_video_by_frames(video_names, new_video_names)


def filter_videos_by_motion(path_videos, path_problematic_videos, PARAM_CORR2D_COEFF):
    list_videos = sorted(glob.glob(path_videos + "*.mp4"), key=os.path.getmtime)
    os.makedirs(path_problematic_videos, exist_ok=True)

    for video in track(list_videos, description="Filtering videos by motion..."):
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
        count = 0
        if np.median(corr_list) >= PARAM_CORR2D_COEFF:
            count += 1
            shutil.move(video, path_problematic_videos + video_name)
    console.print(f"Filtered out {count} videos to {path_problematic_videos}", style="magenta")


def get_all_clips_for_action(output_file):
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

    with open(output_file, 'w+') as fp:
        json.dump(dict_action_clips, fp)


def save_clip_features(image_features, text_features, directories):
    data_dir = 'data/clip_features2/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    assert len(directories) == len(image_features)
    console.print(f"#Saved CLIP features: {len(directories)}", style="magenta")
    for i in track(range(len(directories)), description="Saving CLIP features..."):
        # action, video = directories[i].split("+")[0], "+".join(directories[i].split("+")[1:])
        clip_feature_i = image_features[i]
        text_feature_i = text_features[i]

        torch.save(clip_feature_i, data_dir + directories[i] + "_clip" + '.pt')
        torch.save(text_feature_i, data_dir + directories[i] + "_text" + '.pt')


class VideoDataset(Dataset):
    def __init__(self, directories: Sequence[str], transform: Callable[[Image.Image], torch.Tensor],
                 tokenizer: Callable[[str], torch.Tensor]) -> None:
        super().__init__()
        self.directories = directories
        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i: int) -> Mapping[str, Any]:
        directory = self.directories[i]

        output = {}

        images_per_video = sorted([filename
                                   for filename in os.listdir(directory)
                                   if filename.endswith((".png", ".jpg", ".jpeg"))])
        name = os.path.splitext(images_per_video[0])[0]

        action = " ".join(name.split("+")[0].split("_"))
        output["text_tokens"] = self.tokenizer(f"This is a photo of action {action}")

        output["video"] = [self.transform(Image.open(os.path.join(directory, image_name)))
                           for image_name in images_per_video]

        return output

    def __len__(self) -> int:
        return len(self.directories)


def run_clip(input_file):
    # data_dir = "data/videos/"
    data_dir = "data/videos_sample/"

    with open(input_file) as json_file:
        dict_test_clip = json.load(json_file)

    list_folders_to_process = []
    for action in dict_test_clip:
        for dict_video_time in dict_test_clip[action]:
            video, time_s, time_e = dict_video_time.values()
            video_name = "+".join(["_".join(action.split()), video, time_s, time_e])
            list_folders_to_process.append(video_name)
    model, preprocess = clip.load("ViT-B/16")
    directories = [video_dir for video_dir in [data_dir + folder for folder in list_folders_to_process]]
    # nb_frames = 4
    prompt = "This is a photo of action "
    nb_elem_in_batch = 200
    list_img_features, list_text_features = [], []
    directories_batches = [directories[i:i + nb_elem_in_batch] for i in range(0, len(directories), nb_elem_in_batch)]
    for batch_dir in track(directories_batches, description="Extracting data for CLIP..."):
        prep_images, texts = [], []
        for dir_video in batch_dir:
            if not os.path.exists(dir_video):
                list_folders_to_process.remove(dir_video.replace(data_dir, ''))
                print(f"Path {dir_video} doesn't exist! Skipping ...")
                continue
            images_per_video = sorted(
                [filename for filename in os.listdir(dir_video) if filename.endswith((".png", ".jpeg"))])
            name = os.path.splitext(images_per_video[0])[0]
            action = " ".join(name.split("+")[0].split("_"))
            description = prompt + action
            nb_frames = len(images_per_video)
            for image_name in images_per_video:
                image = Image.open(os.path.join(dir_video, image_name))
                preprocessed_img = preprocess(image)
                prep_images.append(preprocessed_img)
            texts.append(description)

        assert len(prep_images) / nb_frames == len(texts)

        image_input = torch.stack(prep_images).to(DEVICE)
        text_tokens = clip.tokenize(texts).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            text_features = model.encode_text(text_tokens).float()

        # Get the mean of image features for each video
        reshaped_image_features = image_features.reshape(image_features.shape[0] // nb_frames, nb_frames, -1)
        image_features = reshaped_image_features.mean(dim=1)
        assert image_features.shape == text_features.shape

        list_img_features.append(image_features)
        list_text_features.append(text_features)

    image_features = torch.cat(list_img_features)
    text_features = torch.cat(list_text_features)

    # image_features /= image_features.norm(dim=-1, keepdim=True)
    # text_features /= text_features.norm(dim=-1, keepdim=True)

    # similarity = image_features @ text_features.T
    # print(similarity)
    return image_features, text_features, list_folders_to_process


def test_run_clip(input_file):
    data_dir = "data/videos_test/"
    with open(input_file) as json_file:
        dict_test_clip = json.load(json_file)
    list_folders_to_process = []
    for action in dict_test_clip:
        for dict_video_time in dict_test_clip[action]:
            video, time_s, time_e = dict_video_time.values()
            video_name = "+".join(["_".join(action.split()), video, time_s, time_e])
            list_folders_to_process.append(video_name)
    model, preprocess = clip.load("ViT-B/16")
    directories = [video_dir for video_dir in [data_dir + folder for folder in list_folders_to_process]]
    # nb_frames = 4
    prompt = "This is a photo of a person "
    nb_elem_in_batch = 200
    list_img_features, list_text_features = [], []
    directories_batches = [directories[i:i + nb_elem_in_batch] for i in
                           range(0, len(directories), nb_elem_in_batch)]
    for batch_dir in track(directories_batches, description="Extracting data for CLIP..."):
        prep_images, texts = [], []
        for dir_video in batch_dir:
            if not os.path.exists(dir_video):
                list_folders_to_process.remove(dir_video.replace(data_dir, ''))
                # print(f"Path {dir_video} doesn't exist! Skipping ...")
                continue
            images_per_video = sorted(
                [filename for filename in os.listdir(dir_video) if filename.endswith((".png", ".jpeg"))])
            name = os.path.splitext(images_per_video[0])[0]
            action = " ".join(name.split("+")[0].split("_"))
            print(action)
            description = prompt + action
            nb_frames = len(images_per_video)
            for image_name in images_per_video:
                image = Image.open(os.path.join(dir_video, image_name))
                preprocessed_img = preprocess(image)
                prep_images.append(preprocessed_img)
            texts.append(description)

        assert len(prep_images) / nb_frames == len(texts)

        image_input = torch.stack(prep_images).to(DEVICE)
        text_tokens = clip.tokenize([desc for desc in texts]).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            text_features = model.encode_text(text_tokens).float()

        # Get the mean of image features for each video
        reshaped_image_features = image_features.reshape(image_features.shape[0] // nb_frames, nb_frames, -1)
        image_features = reshaped_image_features.mean(dim=1)
        assert image_features.shape == text_features.shape

        list_img_features.append(image_features)
        list_text_features.append(text_features)

    image_features = torch.cat(list_img_features, dim=0)
    text_features = torch.cat(list_text_features, dim=0)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = image_features @ text_features.T
    print(similarity)
    similarity = image_features @ image_features.T
    print(similarity)
    # return image_features, text_features, list_folders_to_process


def stats_videos():
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


def get_video_diff():
    with open('data/dict_action_clips_sample.json') as json_file:
        dict_action_clips = json.load(json_file)

    all_videos = {"+".join([dict["video"], dict["time_s"], dict["time_e"]]) + ".mp4"
                  for values in dict_action_clips.values()
                  for dict in values}
    print(len(all_videos))
    all_videos_downloaded = {video_name.replace('data/videos_sample/', '') for video_name in
                             glob.glob("data/videos_sample/*.mp4") + glob.glob("data/filtered_videos/*.mp4")}
    # all_videos_downloaded = {video_name.replace('data/videos_sample/', '') for video_name in glob.glob("data/videos_sample/*.mp4")}
    print(len(all_videos_downloaded))
    not_downloaded = all_videos - all_videos_downloaded
    print(len(not_downloaded))

    dict_action_clips_sample_remained = defaultdict(list)
    for action, values in dict_action_clips.items():
        for dict in values:
            if "+".join([dict["video"], dict["time_s"], dict["time_e"]]) + ".mp4" in not_downloaded:
                dict_action_clips_sample_remained[action].append(dict)
    with open('data/dict_action_clips_sample_remained.json', 'w+') as fp:
        json.dump(dict_action_clips_sample_remained, fp)


def sample_videos():
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


def test_clip():
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
    # print(cosine_similarity(action1, action2), cosine_similarity(action1, action3), cosine_similarity(action1, action4))

    nodes = pd.read_csv('data/graph/all_visclip_nodes.csv', index_col=0)
    action1 = nodes.loc[['clean sink']].to_numpy()
    action2 = nodes.loc[['scrub sink']].to_numpy()
    action3 = nodes.loc[['clean kitchen']].to_numpy()
    action4 = nodes.loc[['compete with dollar store']].to_numpy()
    action5 = nodes.loc[['put music']].to_numpy()
    action6 = nodes.loc[['add beet']].to_numpy()
    print(cosine_similarity(action1, action2), cosine_similarity(action1, action3), cosine_similarity(action1, action4),
          cosine_similarity(action1, action5), cosine_similarity(action1, action6))

    nodes = pd.read_csv('data/graph/all_weighted_visclip_avg_nodes.csv', index_col=0)
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
    # print(cosine_similarity(action1, action2), cosine_similarity(action1, action3), cosine_similarity(action1, action4))


if __name__ == '__main__':
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
    #
    # image_features, text_features, action_clip_pairs = run_clip(input_file="data/dict_action_clips_sample.json")
    # save_clip_features(image_features, text_features, action_clip_pairs)

    # test_run_clip(input_file="data/dict_action_clips_sample_test.json")
    test_clip()

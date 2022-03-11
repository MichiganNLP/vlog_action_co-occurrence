import glob
import json
import os
import shutil
import clip
import cv2
import ffmpeg
import numpy as np
import torch
from PIL import Image
import subprocess
from rich.progress import track
from rich.console import Console
console = Console()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_video_by_frames(video_names, new_video_names):
    for video, video_new in track(zip(video_names, new_video_names), description="Splitting videos into frames..."):
        print(f"Processing video {video} ...")
        path_in = "data/videos_sample/" + video + ".mp4"
        path_out = "data/videos_sample/" + video_new + "/"
        if not os.path.exists(path_in) or not os.path.exists(path_out):
            print(f"Skipping splitting video by frames: {video_new}")
            continue
        probe = ffmpeg.probe(path_in)
        time = float(probe['streams'][0]['duration']) // 2
        width = probe['streams'][0]['width']

        # Set how many spots you want to extract a video from.
        parts = 4
        intervals = time // parts
        intervals = int(intervals)
        interval_list = [(i * intervals, (i + 1) * intervals) for i in range(parts)]
        i = 0

        if not os.path.exists(path_out):
            os.makedirs(path_out)

        for item in interval_list:
            (
                ffmpeg
                    .input(path_in, ss=item[1])
                    .filter('scale', width, -1)
                    .output(path_out + video_new + "_" + str(i) + '.jpeg', vframes=1)
                    .run()
            )
            i += 1


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
        if np.median(corr_list) >= PARAM_CORR2D_COEFF:
            shutil.move(video, path_problematic_videos + video_name)


def get_all_clips_for_action(output_file):
    with open('data/dict_video_action_pairs_filtered.json') as json_file:
        dict_video_action_pairs_filtered = json.load(json_file)

    dict_action_clips = {}
    for video in dict_video_action_pairs_filtered.keys():
        for [(action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)] in \
                dict_video_action_pairs_filtered[video]:
            if action_1 not in dict_action_clips:
                dict_action_clips[action_1] = []
            [time_s, time_e] = clip_a1
            time_s, time_e = time_s.split(".")[0], time_e.split(".")[0]
            if {"video": video, "time_s": time_s, "time_e": time_e} not in dict_action_clips[action_1]:
                dict_action_clips[action_1].append({"video": video, "time_s": time_s, "time_e": time_e})

            if action_2 not in dict_action_clips:
                dict_action_clips[action_2] = []
            [time_s, time_e] = clip_a2
            time_s, time_e = time_s.split(".")[0], time_e.split(".")[0]
            if {"video": video, "time_s": time_s, "time_e": time_e} not in dict_action_clips[action_2]:
                dict_action_clips[action_2].append({"video": video, "time_s": time_s, "time_e": time_e})

    with open('data/dict_action_clips.json', 'w+') as fp:
        json.dump(dict_action_clips, fp)

    # dict_action_clips_sample = {"put tea into station": dict_action_clips["put tea into station"][:10]}
    # 10 actions, 1 video per action
    dict_action_clips_sample = {action: dict_action_clips[action][:1] for action in list(dict_action_clips.keys())[:10]}

    with open(output_file, 'w+') as fp:
        json.dump(dict_action_clips_sample, fp)
    return dict_action_clips


def save_clip_features(clip_features, text_features, directories):
    data_dir = 'data/clip_features/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for i in track(range(len(directories)), description="Saving CLIP features..."):
        # action, video = directories[i].split("+")[0], "+".join(directories[i].split("+")[1:])
        clip_feature_i = clip_features[i]
        text_feature_i = text_features[i]

        torch.save(clip_feature_i, data_dir + directories[i] + "_clip" + '.pt')
        torch.save(text_feature_i, data_dir + directories[i] + "_text" + '.pt')


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

    model, preprocess = clip.load("ViT-B/32")
    prep_images, texts = [], []
    directories = [video_dir for video_dir in [data_dir + folder for folder in list_folders_to_process]]
    nb_frames = 4
    prompt = "This is a photo of a person "
    for dir_video in track(directories, description="Extracting CLIP features..."):
        images_per_video = sorted(
            [filename for filename in os.listdir(dir_video) if filename.endswith((".png", ".jpeg"))])
        name = os.path.splitext(images_per_video[0])[0]
        action = " ".join(name.split("+")[0].split("_"))
        description = prompt + action
        nb_frames = len(images_per_video)
        for image_name in images_per_video:
            image = Image.open(os.path.join(dir_video, image_name)).convert("RGB")
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

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # similarity = image_features @ text_features.T
    # print(similarity)

    return image_features, text_features, list_folders_to_process

def stats_videos():
    with open('data/coref_all_sentence_transcripts.json') as json_file:
        all_sentence_transcripts_rachel = json.load(json_file)
    nb_videos = len(all_sentence_transcripts_rachel)
    console.print(f"#Unique videos: {nb_videos}", style="magenta")

if __name__ == '__main__':
    pass
    stats_videos()
    # dict_action_clips = get_all_clips_for_action(output_file="data/dict_action_clips_sample.json")
    # subprocess.run(["./download_videos.sh"])
    # filter_videos_by_motion(path_videos="data/videos_sample/", path_problematic_videos="data/filtered_videos/",
    #                         PARAM_CORR2D_COEFF=0.9)
    # split_videos_into_frames(input_file="data/dict_action_clips_sample.json")
    # image_features, text_features, action_clip_pairs = run_clip(input_file="data/dict_action_clips_sample.json")
    # save_clip_features(image_features, text_features, action_clip_pairs)



import json
import os
import ffmpeg
import shutil
import numpy as np
import cv2
import glob
import torch
import clip
from PIL import Image
from tqdm import tqdm


def download_video(video_id):
    url = "https://www.youtube.com/watch?v=" + video_id
    command_save_video = 'youtube-dl --no-check-certificate -f bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4 -v -o ' \
                         + "data/videos/" + video_id + " " + url
    os.system(command_save_video)


def split_video_by_frames(video_names, new_video_names):
    for video, video_new in zip(video_names, new_video_names):
        print(f"Processing video {video} ...")
        path_in = "data/videos/" + video + ".mp4"
        path_out = "data/videos/" + video + "/"
        if not os.path.exists(path_in):
            continue
        if os.path.exists(path_out):
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

        # os.system('/snap/bin/ffmpeg -i ' + path_in + ' -ss "00:00:05" -vframes 1 ' + path_out + video + ".jpeg")  # 1 frame

        ## Output one image every 2 seconds, named out1.png, out2.png, out3.png, etc.
        ## The %02d dictates that the ordinal number of each output image will be formatted using 2 digits.
        # os.system('/snap/bin/ffmpeg -i ' + path_in + ' -vf fps=2 ' + path_out + video + '_%02d.jpeg')
        # os.system('/snap/bin/ffmpeg -i ' + path_in + '  ' + path_out + video + '_%02d.jpeg')


def split_videos_into_frames(input_file):
    with open(input_file) as json_file:
        dict_test_clip = json.load(json_file)
    video_names = []
    new_video_names = []
    for action in dict_test_clip:
        for dict_video_time in dict_test_clip[action]:
            video_name, time_s, time_e = dict_video_time.values()
            video_name = "+".join([video_name, time_s, time_e])
            new_video_name = "+".join(["_".join(action.split()), video_name, time_s, time_e])
            video_names.append(video_name)
            new_video_names.append(new_video_name)

    split_video_by_frames(video_names, new_video_names)


def filter_videos_by_motion(path_videos, path_problematic_videos, PARAM_CORR2D_COEFF):
    list_videos = sorted(glob.glob(path_videos + "*.mp4"), key=os.path.getmtime)
    os.makedirs(path_problematic_videos, exist_ok=True)

    for video in tqdm(list_videos):
        # if "ngYm8nFZJaY+0:00:32+0:00:44" not in video:
        #     continue
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
            # move video in another folder
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

    dict_action_clips_sample = {"put tea into station": dict_action_clips["put tea into station"][:10]}
    with open(output_file, 'w+') as fp:
        json.dump(dict_action_clips_sample, fp)
    return dict_action_clips


def run_CLIP():
    model, preprocess = clip.load("ViT-B/32")

    original_images = []
    prep_images = []
    texts = []
    data_dir = "data/videos"

    directories = [video_name for video_name in os.listdir(data_dir) if ".mp4" not in video_name]
    for dir in directories:
        images = [filename for filename in os.listdir(data_dir + "/" + dir) if filename.endswith(".png") or filename.endswith(".jpeg")]
        name = os.path.splitext(images[0])[0]
        action = " ".join(name.split("+")[0].split("_"))
        description = "This is a photo of a person " + action
        for image_name in images:
            image = Image.open(os.path.join(data_dir + "/" + dir, image_name)).convert("RGB")
            preprocessed_img = preprocess(image)
            break #TODO: process more than 1 frame

        original_images.append(image)
        prep_images.append(preprocessed_img)
        texts.append(description)

    image_input = torch.tensor(np.stack(prep_images)).cuda()
    text_tokens = clip.tokenize([desc for desc in texts]).cuda()

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    print(similarity)

if __name__ == '__main__':
    # dict_action_clips = get_all_clips_for_action(output_file="data/dict_action_clips_sample.json")
    ### run downnload_videos.sh
    # split_videos_into_frames(input_file="data/dict_action_clips_sample.json")
    run_CLIP()

    ################# old
    # # download_video(video_id="zXqBCqPa9VY")

    # filter_videos_by_motion(path_videos="data/videos/", path_problematic_videos="data/filtered_videos/",
    #                         PARAM_CORR2D_COEFF=0.9)

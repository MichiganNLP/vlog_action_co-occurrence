import json
import ast
import os
import numpy as np
import glob
import cv2
import shutil
from tqdm import tqdm

with open('data/final_data/dict_together_actions.json') as json_file:
    dict_together_actions = json.load(json_file)

with open('data/final_data/dict_not_together_actions.json') as json_file:
    dict_not_together_actions = json.load(json_file)


# filter_out_actiontuples_same_video
def prepare_data():
    dict_videos = {"video": []}
    dict_sample_data = {}
    for dict in [dict_together_actions, dict_not_together_actions]:
        for action_tuple in list(dict.keys())[:10]:
            if action_tuple in dict_sample_data:
                raise ValueError("error")
            dict_sample_data[action_tuple] = []
            for video_sentences in dict[action_tuple][:5]:
                (video_tuple, sentence_tuple) = ast.literal_eval(video_sentences)
                if video_tuple[0] == video_tuple[1]:
                    print("Problem: action tuple have exact video", video_tuple[0], action_tuple)
                    print("removing action tuple data point..")
                    continue
                dict_sample_data[action_tuple] += [video_sentences]
                for video in video_tuple:
                    video_name, time_s, time_e = video.split("+")
                    data = {"time_s": time_s, "time_e": time_e, "video": video_name}
                    if data not in dict_videos["video"]:
                        dict_videos["video"].append(data)

    count_total_pairs = 0
    for key in dict_sample_data:
        count_total_pairs += len(dict_sample_data[key])
    print("count_pairs", len(dict_sample_data.keys()))
    print("count_videos", len(dict_videos["video"]))
    print("count_total_data", count_total_pairs)

    with open('data/final_data/dict_sample_for_video.json', 'w+') as fp:
        json.dump(dict_videos, fp)

    with open('data/final_data/dict_sample_data.json', 'w+') as fp:
        json.dump(dict_sample_data, fp)

def format_text_info():
    sample_data = {"together": [], "not_together": []}
    for dict in [dict_together_actions, dict_not_together_actions]:
        d = {}
        for action_tuple in list(dict.keys()):
            d[action_tuple] = []
            for video_sentences in dict[action_tuple]:
                (video_tuple, sentence_tuple) = ast.literal_eval(video_sentences)
                if video_tuple[0] == video_tuple[1]:
                    # print("Problem: action tuple have exact video", video_tuple[0], action_tuple)
                    # print("removing action tuple data point..")
                    continue
                d[action_tuple] += [video_sentences]
        if dict == dict_together_actions:
            sample_data["together"] = d
        else:
            sample_data["not_together"] = d

    with open('data/final_data/dict_format_data.json', 'w+') as fp:
        json.dump(sample_data, fp)


def filter_videos_by_motion(PATH_videos, PATH_problematic_videos, PARAM_CORR2D_COEFF):
    list_videos = sorted(glob.glob(PATH_videos + "*.mp4"), key=os.path.getmtime)
    os.makedirs(PATH_problematic_videos, exist_ok=True)

    for video in tqdm(list_videos):
        # if "ngYm8nFZJaY+0:00:32+0:00:44" not in video:
        #     continue
        vidcap = cv2.VideoCapture(video)
        if (vidcap.isOpened() == False):
            continue
        corr_list = []
        video_name = video.split("/")[-1]
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_nb_1 in range(0, length - 100, 100):
            vidcap.set(1, frame_nb_1)
            success, image = vidcap.read()
            if success == False:
                continue
            gray_image_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            frame_nb_2 = frame_nb_1 + 100
            vidcap.set(1, frame_nb_2)
            success, image = vidcap.read()
            if success == False:
                continue
            gray_image_2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            corr2_matrix = np.corrcoef(gray_image_1.reshape(-1), gray_image_2.reshape(-1))
            corr2 = corr2_matrix[0][1]
            corr_list.append(corr2)

        # print(video_name, np.median(corr_list))
        if np.median(corr_list) >= PARAM_CORR2D_COEFF:
            # move video in another folder
            shutil.move(video, PATH_problematic_videos + video_name)



def main():
    # format_text_info()
    # prepare_data()
    filter_videos_by_motion(PATH_videos="data/videos/", PATH_problematic_videos="data/filtered_videos/",
                            PARAM_CORR2D_COEFF=0.9)


if __name__ == '__main__':
    main()

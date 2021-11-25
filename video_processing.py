import json
import os
import ffmpeg

def download_video(video_id):
    url = "https://www.youtube.com/watch?v=" + video_id
    command_save_video = 'youtube-dl --no-check-certificate -f bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4 -v -o ' \
                         + "data/videos/" + video_id + " " + url
    os.system(command_save_video)

def split_video_by_frames(video_names):
    for video in video_names:
        path_in = "data/videos/" + video + ".mp4"
        path_out = "data/videos/" + video + "/"
        print(path_in)
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
                    .output(path_out + video + "_" + str(i) + '.jpeg', vframes=1)
                    .run()
            )
            i += 1

        # os.system('/snap/bin/ffmpeg -i ' + path_in + ' -ss "00:00:05" -vframes 1 ' + path_out + video + ".jpeg")  # 1 frame

        ## Output one image every 2 seconds, named out1.png, out2.png, out3.png, etc.
        ## The %02d dictates that the ordinal number of each output image will be formatted using 2 digits.
        # os.system('/snap/bin/ffmpeg -i ' + path_in + ' -vf fps=2 ' + path_out + video + '_%02d.jpeg')
        # os.system('/snap/bin/ffmpeg -i ' + path_in + '  ' + path_out + video + '_%02d.jpeg')

def split_videos_into_frames():
    # with open('data/analyse_verbs/dict_test_clip.json') as json_file:
    #     dict_test_clip = json.load(json_file)
    # video_names = [v[:-4] for v in dict_test_clip.keys()]

    list_videos = ["4ES4nNtbcNU", "cQCLMbOmEAU", "yl6vBuP23bE", "ZO88Cj_hjQk"]
    with open('data/analyse_verbs/dict_example3_actions.json') as json_file:
        dict_test_clip = json.load(json_file)
    video_names = []
    for key in dict_test_clip:
        for video in list_videos:
            if video in key:
                video_name, time_s, time_e = key[:-4].split("+")
                new_video_name = "+".join([video_name, time_s.split(".")[0], time_e.split(".")[0]])
                video_names.append(new_video_name)

    # print(video_names)
    split_video_by_frames(video_names)

def make_test_clip():
    dict_test_clip = {}
    with open('data/analyse_verbs/dict_example3_actions.json') as json_file:
        dict_example3_actions = json.load(json_file)
    list_videos = ["4ES4nNtbcNU", "cQCLMbOmEAU", "yl6vBuP23bE", "ZO88Cj_hjQk"]

    for key in dict_example3_actions:
        for video in list_videos:
            if video in key:
                video_name, time_s, time_e = key[:-4].split("+")
                new_video_name = "+".join([video_name, time_s.split(".")[0], time_e.split(".")[0]])
                if new_video_name not in dict_test_clip:
                    dict_test_clip[new_video_name] = []
                for data in dict_example3_actions[key]:
                    dict_test_clip[new_video_name].append({"action":data["action"], "sentence":data["sentence"]})

    with open('data/analyse_verbs/dict_test_clip2.json', 'w+') as fp:
        json.dump(dict_test_clip, fp)


# def split_video_by_time(video_id, time_start, time_end, github_path):
#     duration = time_end - time_start
#     print(time_start)
#     print(time_end)
#     print(duration)
#     time_start = str(datetime.timedelta(seconds=time_start))
#     time_end = str(datetime.timedelta(seconds=time_end))
#     duration = str(datetime.timedelta(seconds=duration))
#     print(time_start)
#     print(time_end)
#     path_video = 'data/videos/' + video_id + '.mp4 '
#     command_split_video = 'ffmpeg -ss ' + time_start + ' -i ' + path_video + "-fs 25M " + '-to ' + duration + \
#                           ' -c copy ' + github_path + video_id + '+' + time_start + '+' + time_end + '.mp4'
#
#     print(command_split_video)
#     os.system(command_split_video)

def format_file():
    # cQCLMbOmEAU
    list_videos = ["4ES4nNtbcNU", "cQCLMbOmEAU", "yl6vBuP23bE", "ZO88Cj_hjQk"]
    with open('data/analyse_verbs/dict_example3_actions.json') as json_file:
        dict_example3 = json.load(json_file)
    
    dict = {"video": []}
    for key in dict_example3:
        for video in list_videos:
            if video in key:
                video, time_s, time_e = key.split("+")
                dict["video"].append({"time_s": time_s, "time_e": time_e[:-4], "video": video})

    with open('data/analyse_verbs/dict_example3_for_video.json', 'w+') as fp:
        json.dump(dict, fp)
        
        
if __name__ == '__main__':
    # download_video(video_id="zXqBCqPa9VY")
    # split_videos_into_frames()
    make_test_clip()
    # format_file()
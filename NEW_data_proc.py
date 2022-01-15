import time
import spacy
import itertools
import datetime
import json
import ast
from tqdm import tqdm
from collections import Counter
import en_core_web_sm
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
import numpy as np

nlp = en_core_web_sm.load()

def get_all_verb_dobj(sentence, coref_sentence, list_mentions):
    verbs = ["add", "shower", "sleep", "eat", "make", "clean", "dance", "sing", "drink", "buy", "drive",
             "cook", "fall", "read", "write", "study", "play", "paint", "listen", "relax", "wake",
             "walk", "shop", "work", "learn", "cut", "brush", "wash", "run", "throw", "paint", "cool",
             "warm", "give", "use"]
    tokens = nlp(coref_sentence)
    # list_mentions = ast.literal_eval(list_mentions)
    list_actions = []
    for t in tokens:
        # if t.pos_ == "VERB" and t.lemma_ in verbs:
        if t.pos_ == "VERB":
        # if t.pos_ == "VERB":
            # if t.lemma_ not in ["have", "ve", "do", "can", "will", "could", "would", "have", "share", "welcome", "am",
            #                     "be", "was", "were", "let", "love", "like", "say", "tell"]:
            action = t.lemma_
            for tok in t.children:
                dep = tok.dep_
                # if tok.lemma_ in ['that', 'this', 'them', 'these', 'those']:
                #     children = tok.children
                #     for child in children:
                #         print(child.lemma_, child.dep_)
                if dep == "dobj" or dep == "prt" or dep == "xcomp":
                    action += (" " + tok.text)

            # if list_mentions == "[]":
            if len(action.split()) >= 2:
                list_actions.append(action)
            # else:
            #     print(action, list_mentions)

    return list_actions

def get_verbs_all_videos():
    with open('data/coref_all_sentence_transcripts.json') as json_file:
        # with open('data/UCM3P_G21gOSVdepXrEFojIg.json') as json_file:
        all_sentence_transcripts_rachel = json.load(json_file)

    dict_verb_dobj_per_video = {}
    dict_verb_dobj = {}
    print(len(all_sentence_transcripts_rachel.keys()))
    for video in tqdm(list(all_sentence_transcripts_rachel.keys())[:100]):
        actions_per_video = []
        for dict in list(all_sentence_transcripts_rachel[video]):
            part_sentence, coref_sentence, time_s, time_e, list_mentions = dict["sentence"], dict["coref_sentence"], \
                                                                           dict["time_s"], dict["time_e"], dict[
                                                                               "list_mentions"]
            # list_verbs = get_all_verbs(part_sentence)
            list_actions = list(set(get_all_verb_dobj(part_sentence, coref_sentence, list_mentions)))
            if list_actions:
                for action in list_actions:
                    time_start = str(datetime.timedelta(seconds=time_s))
                    time_end = str(datetime.timedelta(seconds=time_e))
                    actions_per_video.append(str([action, part_sentence, [time_start, time_end]]))
                    # actions_per_video.append([action, part_sentence, [time_start, time_end]])
        dict_verb_dobj_per_video[video] = actions_per_video

    #         for verb in list_verbs:
    #             verbs_per_video.append(verb)
    #     dict_verbs_per_video[video] = verbs_per_video
    #
    # with open('data/analyse_verbs/dict_verbs_per_video_rachel.json', 'w+') as fp:
    #     json.dump(dict_verbs_per_video, fp)
    # with open('data/analyse_verbs/dict_verb_dobj_per_video_all_coref.json', 'w+') as fp:
    with open('data/NEW/dict_verb_dobj_per_video_all_coref_verbs.json', 'w+') as fp:
        json.dump(dict_verb_dobj_per_video, fp)

def get_most_common_actions():
    with open('data/NEW/dict_verb_dobj_per_video_all_coref_verbs.json') as json_file:
        dict_verb_dobj_per_video = json.load(json_file)
    all_actions = []
    for video in dict_verb_dobj_per_video:
        for action_data in dict_verb_dobj_per_video[video]:
            all_actions.append(ast.literal_eval(action_data)[0])

    threshold = 2
    most_common_actions = []
    for a_c in Counter(all_actions).most_common():
        if a_c[1] >= threshold:
            most_common_actions.append(a_c[0])
    return most_common_actions

def cluster_actions(list_actions):
    model = SentenceTransformer(
        'stsb-roberta-base')  # models: https://www.sbert.net/docs/predeved_models.html#semantic-textual-similarity
    list_embeddings = model.encode(list_actions, show_progress_bar=True, convert_to_tensor=True)

    print("Start clustering")
    start_time = time.time()

    # Normalize the embeddings to unit length
    corpus_embeddings = list_embeddings / np.linalg.norm(list_embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=1.5)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(list_actions[sentence_id])

    dict_clustered_actions = {}
    for i, cluster in clustered_sentences.items():
        print("Cluster ", i + 1)
        print(cluster)
        print("")
        dict_clustered_actions[cluster[0]] = cluster[1:]
    print(len(clustered_sentences.items()))
    print(len(dict_clustered_actions))
    return dict_clustered_actions

def get_clustered_action_name(dict_clustered_actions, action):
    if action in dict_clustered_actions:
        return action
    for action_key in dict_clustered_actions:
        values = dict_clustered_actions[action_key]
        if action in values:
            return action_key
    raise ValueError(action + "not in dict_clustered_actions")


def get_action_pairs_co_occuring_per_video(most_common_actions, dict_clustered_actions):
    threshold_utterances = 10  # x utterances or time (30 sec) TODO (- SET threshold)
    with open('data/NEW/dict_verb_dobj_per_video_all_coref_verbs.json') as json_file:
        dict_verb_dobj_per_video = json.load(json_file)

    list_all_action_combinations = []
    for video in tqdm(dict_verb_dobj_per_video):
        list_all_actions = []
        # list_all_transcript = []
        # list_all_video = []
        for action_data in dict_verb_dobj_per_video[video]:
            action_data = ast.literal_eval(action_data)
            action, _, _ = action_data
            list_all_actions.append(action)

        list_action_chunks = [list_all_actions[i:i + threshold_utterances] for i in range(len(list_all_actions))]
        list_action_combinations = [pair
                                    for action_chunks in list_action_chunks
                                    for pair in itertools.combinations(action_chunks, 2)]

        list_all_action_combinations += [action_pair for action_pair in list_action_combinations]

    #TODO (- SET threshold): remove action pairs which don't appear more than threshold times and that don't contain most common actions
    threshold = 10
    most_common_pairs = []
    for a_c in Counter(list_all_action_combinations).most_common():
        # if a_c[1] >= threshold:
        if a_c[1] >= threshold and a_c[0][0] != a_c[0][1] and a_c[0][0] in most_common_actions and a_c[0][1] in most_common_actions:
            most_common_pairs.append(a_c[0])

    print("From initial {0} all action combination, after filtering {1}".format(len(list_all_action_combinations),
                                                                                len(most_common_pairs)))

    dict_video_action_pairs = {}
    for video in tqdm(dict_verb_dobj_per_video):
        dict_video_action_pairs[video] = []
        list_all_actions, list_all_transcripts, list_all_clips = [], [], []
        for action_data in dict_verb_dobj_per_video[video]:
            action_data = ast.literal_eval(action_data)
            action, transcript, clip = action_data
            list_all_actions.append(action)
            list_all_transcripts.append(transcript)
            list_all_clips.append(clip)
        list_action_chunks = [list_all_actions[i:i + threshold_utterances] for i in range(len(list_all_actions))]
        list_transcript_chunks = [list_all_transcripts[i:i + threshold_utterances] for i in range(len(list_all_transcripts))]
        list_clip_chunks = [list_all_clips[i:i + threshold_utterances] for i in range(len(list_all_clips))]

        cluster_fitered_actions = []
        for (a1, a2) in most_common_pairs:
            for idx, chunk in enumerate(list_action_chunks):
                if a1 in chunk and a2 in chunk:
                    transcript_chunk = list_transcript_chunks[idx]
                    clip_chunk = list_clip_chunks[idx]
                    a1_idx = chunk.index(a1)
                    a2_idx = chunk.index(a2)
                    transcript_a1, clip_a1 = transcript_chunk[a1_idx], clip_chunk[a1_idx]
                    transcript_a2, clip_a2 = transcript_chunk[a2_idx], clip_chunk[a2_idx]
                    # rename actions to their clusters
                    a1_clustered = get_clustered_action_name(dict_clustered_actions, a1)
                    a2_clustered = get_clustered_action_name(dict_clustered_actions, a2)
                    if a1_clustered != a2_clustered: #TODO: MIGHT NOT NEED THIS
                        # dict_video_action_pairs[video].append([(a1, transcript_a1, clip_a1), (a2, transcript_a2, clip_a2)])
                        dict_video_action_pairs[video].append([(a1_clustered, transcript_a1, clip_a1), (a2_clustered, transcript_a2, clip_a2)])
                        cluster_fitered_actions.append(a1_clustered)
                        cluster_fitered_actions.append(a2_clustered)
                        break
    print(dict_video_action_pairs)
    print("After clustering, there are {0} unique actions and {1} all ".format(len(list(set(cluster_fitered_actions))), len(cluster_fitered_actions)))
    return dict_video_action_pairs



def main():
    # get_verbs_all_videos()
    most_common_actions = get_most_common_actions()
    dict_clustered_actions = cluster_actions(most_common_actions)
    get_action_pairs_co_occuring_per_video(most_common_actions, dict_clustered_actions)


if __name__ == '__main__':
    main()
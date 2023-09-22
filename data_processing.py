import ast
import datetime
import json
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from pyvis.network import Network
from rich.progress import track
from rich.console import Console
import torch
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
console = Console()

# nlp = spacy.load("en_core_web_sm") # if want to run faster, but not as accurate
nlp = spacy.load("en_core_web_trf")


def process_imsitu():
    with open('data/utils/imsitu.json') as json_file:
        dict_imsitu = json.load(json_file)

    verbs = dict_imsitu["verbs"].keys()
    lemmatized_verbs = [lemmatizer.lemmatize(verb, 'v') for verb in verbs]
    return lemmatized_verbs


def get_action_verbs():
    print("Getting all the verbs ...")
    imsitu_verbs = process_imsitu()

    # http://www-personal.umich.edu/~jlawler/levin.verbs
    with open('data/utils/dict_levin_verbs.json') as json_file:
        levin_verbs_dict = json.load(json_file)
    levin_verbs = list(set([v for values in levin_verbs_dict.values() for v in values.split()]))

    with open('data/utils/visual_verbnet_beta2015.json') as json_file:
        visual_verbnet_beta2015 = json.load(json_file)
    list_verbnet_verbs = [data["name"].split("_")[0] for data in visual_verbnet_beta2015["visual_actions"]
                          if data["category"] not in ["communication", "perception", "posture"]]

    verbs_to_remove = ["follow", "like", "exchange", "follow", "show", "subscribe", "be", "show", "reach", "laugh",
                       "avoid", "get", "feed",
                       "run", "carry", "give", "make", "show", "give", "click", "turn", "recommend", "link",
                       "come", "donate", "help", "hang", "hit", "stream", "kick", "blink", "donate", "bump", "play",
                       "beg", "record", "giggle", "speak", "confront", "offer", "attach", "throw", "ignore", "help",
                       "unlock", "encourage", "press", "ask", "distribute", "talk", "display", "say", "jump"]
    levin_verbs = [v for v in levin_verbs if v not in verbs_to_remove]
    verbnet_verbs = [v for v in list_verbnet_verbs if v not in verbs_to_remove]
    imsitu_verbs = [v for v in imsitu_verbs if v not in verbs_to_remove]

    # combine Levin & VisualVerbNet verbs
    console.print(
        f"There are {len(levin_verbs)} Levin and {len(verbnet_verbs)} VisualVerbNet verbs and {len(imsitu_verbs)} ImSitu verbs.",
        style="magenta")
    verbs = list(set(verbnet_verbs + levin_verbs + imsitu_verbs))
    console.print(f"There are {len(verbs)} verbs in total.", style="magenta")

    return verbs


def get_VP(t, action):
    for tok in t.children:
        if tok.dep_ in ["dobj", "prep", "pobj", "compound"]:
            if tok.pos_ in ['ADP', 'NOUN', 'PROPN']:
                action = action + [tok]
            action = get_VP(tok, action)
    return action


def get_verb_dobj(coref_sentences, verbs):
    remove_words = ["some", "mine", "ooh", "thing", "lot", "whatever", "whichever", "much", "many",
                    "most", "bunch", "all", "one", "do", "one", "kind", "it", "a", "thumbs", "what", "which",
                    "who", "whom", "I", "him", "her", "something", "anything", "element",
                    "that", "those", "them", "this", "these", "sense", "be", "decision", "subscribe", "like", "love",
                    "comment", "they", "link", "same", "different", "everything", "nothing", "button", "video",
                    "button"]

    list_actions = []
    list_not_caught_verbs = []
    for doc in nlp.pipe(coref_sentences):
        list_actions = []
        for t in doc:
            if t.pos_ == "VERB" and t.lemma_ not in verbs:
                list_not_caught_verbs.append(t.lemma_)
            if t.pos_ == "VERB" and t.lemma_ in verbs:
                action = get_VP(t, [t])
                sorted_all = [(tok.lemma_, tok.pos_) for tok in sorted(action, key=lambda tok: tok.i) if
                              tok.lemma_ not in remove_words and tok.lemma_.isalpha() \
                              and not any(letter.isupper() for letter in tok.lemma_)]

                # remove everything before verb
                if not sorted_all or (t.lemma_, 'VERB') not in sorted_all:
                    continue
                if sorted_all[0][0] != t.lemma_:
                    idx_verb = sorted_all.index((t.lemma_, 'VERB'))
                    sorted_all = sorted_all[idx_verb:]
                if len(sorted_all) < 2:
                    continue

                # get only one ADP
                sorted_words = []
                count = 0
                for index, (word, pos) in enumerate(sorted_all):
                    if pos == 'ADP':
                        count += 1
                        if count > 1:
                            break
                    sorted_words.append((word, pos))

                # remove if last is 'ADP'
                if sorted_words[-1][1] == 'ADP':
                    sorted_words = [elem[0] for elem in sorted_words[:-1]]
                else:
                    sorted_words = [elem[0] for elem in sorted_words]

                # remove duplicated words
                sorted_words_no_duplicates = []
                [sorted_words_no_duplicates.append(x) for x in sorted_words if
                 x not in sorted_words_no_duplicates]
                if len(sorted_words_no_duplicates) >= 2:
                    list_actions.append(" ".join(sorted_words_no_duplicates))

        list_actions.append(list_actions)

    return list_actions, list_not_caught_verbs


def get_clustered_action_name(dict_clustered_actions, action):
    if action in dict_clustered_actions:
        return action
    for action_key in dict_clustered_actions:
        values = dict_clustered_actions[action_key]
        if action in values:
            return action_key
    raise ValueError(action + "not in dict_clustered_actions")


def get_diff_time(clip_a1, clip_a2):
    clip_a1 = [clip_a1[0].split(".")[0], clip_a1[1].split(".")[0]]
    clip_a2 = [clip_a2[0].split(".")[0], clip_a2[1].split(".")[0]]
    time1 = datetime.datetime.strptime(clip_a1[1], '%H:%M:%S')
    time2 = datetime.datetime.strptime(clip_a2[0], '%H:%M:%S')
    difference = (time2 - time1).total_seconds()
    print(time2, time1, difference)
    return difference


def show_graph_actions(clustered_action_pairs, video):
    clustered_action_pairs = [action_pair for action_pair in clustered_action_pairs]

    counter = Counter([tuple(x) for x in clustered_action_pairs])
    list_tuples_actions = [(action_pair[0], action_pair[1], counter[action_pair]) for action_pair in counter]
    df = pd.DataFrame(list_tuples_actions, columns=['action1', 'action2', 'nb_examples'])
    sources = df['action1']
    targets = df['action2']
    weights = df['nb_examples']

    net = Network(height='100%', width='100%', bgcolor='#222222', font_color='white')
    # set the physics layout of the network
    net.hrepulsion()

    for index, (src, dst, w) in enumerate(zip(sources, targets, weights)):
        net.add_node(src, label=src, title=str(index), value=100)
        net.add_node(dst, label=dst, title=str(index), value=100)
        net.add_edge(src, dst, label=str(index), title=str(index) + ": " + src + " , " + dst, value=w)

    net.show_buttons(filter_=['physics'])
    net.show("data/graph_plots/" + video + '.html')


def get_actions(verbs, video_sample, try_per_video):
    with open('data/coref_all_sentence_transcripts.json') as json_file:
        sentence_transcripts = json.load(json_file)

    dict_verb_dobj_per_video = {}
    wordsToBeRemoved = ["your", "some", "few", "bit of", "slice", "pinch", "little", "lot", "much", "many", "more",
                        "about"]

    for video in track(sentence_transcripts, description="Getting all actions..."):
        if try_per_video and video != video_sample:
            continue
        coref_sentences, time_start, time_end, sentences = [], [], [], []
        for dict in sentence_transcripts[video]:
            sentence, coref_sentence, time_s, time_e, list_mentions = dict["sentence"], dict["coref_sentence"], \
                dict["time_s"], dict["time_e"], dict[
                "list_mentions"]

            for w in wordsToBeRemoved:
                coref_sentence = coref_sentence.replace(" " + w + " ", " ")

            coref_sentences.append(coref_sentence)
            time_start.append(str(datetime.timedelta(seconds=time_s)))
            time_end.append(str(datetime.timedelta(seconds=time_e)))
            sentences.append(sentence)

        actions_per_video, list_not_caught_verbs = get_verb_dobj(coref_sentences, verbs)

        for (actions_per_sentence, time_start, time_end, sentence) in zip(actions_per_video, time_start, time_end,
                                                                          sentences):
            if actions_per_sentence:
                actions_per_video += [str([action, sentence, [time_start, time_end]]) for action in
                                      actions_per_sentence]
        dict_verb_dobj_per_video[video] = actions_per_video

    with open('data/dict_video_actions.json', 'w+') as fp:
        json.dump(dict_verb_dobj_per_video, fp)


def get_action_pairs_by_time(time_difference):
    with open('data/dict_renamed_actions.json') as json_file:
        dict_verb_dobj_per_video = json.load(json_file)

    dict_video_action_pairs = {}
    actions = []
    for video in track(dict_verb_dobj_per_video, description=f"Getting action pairs by time diff {time_difference}..."):
        dict_video_action_pairs[video] = []
        actions_time = [ast.literal_eval(action_data) for action_data in dict_verb_dobj_per_video[video]]
        for i in range(0, len(actions_time) - 1):
            for j in range(i + 1, len(actions_time)):
                actions.append(actions_time[i][0])
                difference = get_diff_time(actions_time[i][2], actions_time[j][2])
                if difference > time_difference:
                    break
                action_1, transcript_a1, clip_a1 = actions_time[i]
                action_2, transcript_a2, clip_a2 = actions_time[j]
                if action_1 != action_2 and [(action_1, transcript_a1, clip_a1),
                                             (action_2, transcript_a2, clip_a2)] not in dict_video_action_pairs[video] \
                        and [(action_2, transcript_a2, clip_a2), (action_1, transcript_a1, clip_a1)] not in \
                        dict_video_action_pairs[video]:
                    dict_video_action_pairs[video].append(
                        [(action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)])
    console.print(f"#Unique actions before co-occurrence: {len(set(actions))}", style="magenta")
    with open('data/dict_video_action_pairs.json', 'w+') as fp:
        json.dump(dict_video_action_pairs, fp)
    return dict_video_action_pairs


def plot_graph_actions(video, input):
    with open(input) as json_file:
        dict_video_action_pairs = json.load(json_file)
    action_pairs = [sorted((action_1, action_2))
                    for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
                    in dict_video_action_pairs[video]]
    show_graph_actions(action_pairs, video)


def get_stats_actions(before_clustering):
    if before_clustering:
        with open('data/dict_video_action_pairs.json') as json_file:
            dict_video_action_pairs = json.load(json_file)
    else:
        with open('data/dict_video_action_pairs_filtered_by_cluster.json') as json_file:
            dict_video_action_pairs = json.load(json_file)

    verbs, actions, action_pairs = [], [], []
    for video in dict_video_action_pairs:
        for [(action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)] in dict_video_action_pairs[video]:
            actions.append(action_1)
            actions.append(action_2)
            action_pairs.append(str(sorted((action_1, action_2))))
            verbs.append(action_1.split()[0])
            verbs.append(action_2.split()[0])

    word = "before" if before_clustering else "after"
    console.print(f"#Unique actions after co-occurrence, {word} clustering: {len(set(actions))}", style="magenta")
    console.print(f"#Unique action pairs after co-occurrence, {word} clustering: {len(set(action_pairs))}",
                  style="magenta")
    console.print(f"#unique verbs after co-occurrence, {word} clustering: {len(set(verbs))}", style="magenta")
    return set(verbs)


def cluster_actions():
    with open('data/dict_video_action_pairs.json') as json_file:
        dict_video_action_pairs = json.load(json_file)
    actions = []
    for video in dict_video_action_pairs:
        for [(action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)] in dict_video_action_pairs[video]:
            actions.append(action_1)
            actions.append(action_2)

    list_actions = list(set(actions))
    print(f"Clustering {len(list_actions)} unique actions ...")
    model = SentenceTransformer('stsb-roberta-base')  # 'all-MiniLM-L6-v2'
    list_embeddings = model.encode(list_actions, show_progress_bar=True, convert_to_tensor=True)

    print("Start clustering")

    # Normalize the embeddings to unit length
    corpus_embeddings = list_embeddings.cpu() / np.linalg.norm(list_embeddings.cpu(), axis=1, keepdims=True)

    # Perform agglomerative clustering
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=1.5)
    # clustering_model = KMeans(n_clusters=?)

    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
        clustered_sentences[cluster_id].append(list_actions[sentence_id])

    # A higher Silhouette Coefficient score relates to a model with better defined clusters
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(corpus_embeddings, cluster_assignment))
    # ratio between the within-cluster dispersion and the between-cluster dispersion
    # The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
    print("Calinski and Harabasz Score: %0.3f" % metrics.calinski_harabasz_score(corpus_embeddings, cluster_assignment))
    # Zero is the lowest possible score. Values closer to zero indicate a better partition (better separation between the clusters.)
    print("Davies-Bouldin Index: %0.3f" % metrics.davies_bouldin_score(corpus_embeddings, cluster_assignment))

    # assign name of cluster - the most common action from cluster
    dict_clustered_actions = {}
    occurrences = Counter(actions)
    for i, cluster in clustered_sentences.items():
        max_freq = 0
        name_cluster = ""
        for action in cluster:
            freq = occurrences[action]
            if freq > max_freq:
                max_freq = freq
                name_cluster = action
        dict_clustered_actions[name_cluster] = cluster

    console.print(f"#clusters, before filtering: {len(dict_clustered_actions)} ", style="magenta")
    with open('data/dict_clustered_actions.json', 'w+') as fp:
        json.dump(dict_clustered_actions, fp)


def filter_clusters_by_size():
    cluster_size = 2
    print(f"Filtering clusters by size {cluster_size} ...")
    with open('data/dict_clustered_actions.json') as json_file:
        dict_clustered_actions = json.load(json_file)

    filtered_clusters = {}
    count = 0
    for key in dict_clustered_actions:
        if len(dict_clustered_actions[key]) < cluster_size:
            count += 1
        else:
            filtered_clusters[key] = dict_clustered_actions[key]

    print(f"Initial #clusters {len(dict_clustered_actions)}")
    print(f"Removed {count} clusters")
    print(f"Remained {len(filtered_clusters)} clusters")
    return filtered_clusters


# function to return key for any value
def get_key(my_dict, val):
    return next((k for k, v in my_dict.items() if val in v), None)


def filter_pairs_by_cluster(filtered_clusters, TIME_DIFF):
    with open(f'data/dict_video_action_pairs.json') as json_file:
        dict_video_action_pairs = json.load(json_file)
    dict_video_action_pairs_filtered = {}
    for video in track(dict_video_action_pairs, description="Filtering action pairs by clusters ..."):
        if video not in dict_video_action_pairs_filtered:
            dict_video_action_pairs_filtered[video] = []
        for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2) in dict_video_action_pairs[video]:
            if action_1 in filtered_clusters and action_2 in filtered_clusters:
                dict_video_action_pairs_filtered[video].append(
                    [(action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)])
            elif action_1 not in filtered_clusters and action_2 in filtered_clusters:
                action_1_key = get_key(filtered_clusters, action_1)
                if action_1_key and action_1_key != action_2:
                    dict_video_action_pairs_filtered[video].append(
                        [(action_1_key, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)])
            elif action_2 not in filtered_clusters and action_1 in filtered_clusters:
                action_2_key = get_key(filtered_clusters, action_2)
                if action_2_key and action_2_key != action_1:
                    dict_video_action_pairs_filtered[video].append(
                        [(action_1, transcript_a1, clip_a1), (action_2_key, transcript_a2, clip_a2)])
            elif action_1 not in filtered_clusters and action_2 not in filtered_clusters:
                action_1_key = get_key(filtered_clusters, action_1)
                action_2_key = get_key(filtered_clusters, action_2)
                if action_1_key and action_2_key and action_1_key != action_2_key:
                    dict_video_action_pairs_filtered[video].append(
                        [(action_1_key, transcript_a1, clip_a1), (action_2_key, transcript_a2, clip_a2)])

    # with open(f'data/dict_video_action_pairs_filtered_by_cluster_{TIME_DIFF}.json', 'w+') as fp:
    with open(f'data/dict_video_action_pairs_filtered_by_cluster.json', 'w+') as fp:
        json.dump(dict_video_action_pairs_filtered, fp)


def combine_graphs(video_sample, TIME_DIFF, directed, sample, filter_by_link):
    # with open(f'data/dict_video_action_pairs_filtered_by_cluster_{TIME_DIFF}.json') as json_file:
    with open(f'data/dict_video_action_pairs_filtered_by_cluster.json') as json_file:
        dict_video_action_pairs_filtered = json.load(json_file)
    transcripts_per_action = {}
    if sample:
        if directed:
            action_pairs = [str((action_1, action_2))
                            for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
                            in dict_video_action_pairs_filtered[video_sample]]
        else:
            action_pairs = [str(sorted((action_1, action_2)))
                            for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
                            in dict_video_action_pairs_filtered[video_sample]]
    else:
        if directed:
            action_pairs = [str((action_1, action_2))
                            for video in dict_video_action_pairs_filtered
                            for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
                            in dict_video_action_pairs_filtered[video]]
        else:
            action_pairs = [str(sorted((action_1, action_2)))
                            for video in dict_video_action_pairs_filtered
                            for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
                            in dict_video_action_pairs_filtered[video]]

    if filter_by_link:
        counter = Counter(action_pairs)
        print("Removing links that appear only once ...")
        action_pairs = [action_pair for action_pair in action_pairs if counter[action_pair] > 1]
        console.print(f"After edge filtering, {len(set(action_pairs))} unique action pairs", style="magenta")
        dict_video_action_pairs_filtered_by_link = {}

        if directed:
            for video in dict_video_action_pairs_filtered:
                for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2) in \
                        dict_video_action_pairs_filtered[video]:
                    if str((action_1, action_2)) in action_pairs:
                        if video not in dict_video_action_pairs_filtered_by_link:
                            dict_video_action_pairs_filtered_by_link[video] = []
                        dict_video_action_pairs_filtered_by_link[video].append(
                            [(action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)])
                        transcripts_per_action[action_1] = transcript_a1
                        transcripts_per_action[action_2] = transcript_a2

        else:
            for video in dict_video_action_pairs_filtered:
                for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2) in \
                        dict_video_action_pairs_filtered[video]:
                    if str(sorted((action_1, action_2))) in action_pairs:
                        if video not in dict_video_action_pairs_filtered_by_link:
                            dict_video_action_pairs_filtered_by_link[video] = []
                        dict_video_action_pairs_filtered_by_link[video].append(
                            [(action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)])
                        transcripts_per_action[action_1] = transcript_a1
                        transcripts_per_action[action_2] = transcript_a2

        # with open(f'data/dict_video_action_pairs_filtered_by_link_{TIME_DIFF}.json', 'w+') as fp:
        with open(f'data/dict_video_action_pairs_filtered_by_link.json', 'w+') as fp:
            json.dump(dict_video_action_pairs_filtered_by_link, fp)

    action_pairs_converted = [ast.literal_eval(action_pair) for action_pair in action_pairs]
    actions, verbs = [], []
    for (a1, a2) in action_pairs_converted:
        actions.append(a1)
        actions.append(a2)
        verbs.append(a1.split()[0])
        verbs.append(a2.split()[0])
    console.print(f"After filtering, {len(set(actions))} unique actions", style="magenta")
    console.print(f"After filtering, {len(set(verbs))} unique verbs", style="magenta")

    return action_pairs, transcripts_per_action

def read_data_train_test(dataset):
    data_train, data_test = set(), set()
    with open(f'data/dict_video_action_pairs_{dataset}.json') as json_file:
        dict_video_action_pairs_COIN = json.load(json_file)
    action_pairs = [sorted((action_1, action_2))
                    for video in dict_video_action_pairs_COIN
                    for (action_1, clip_a1), (action_2, clip_a2)
                    in dict_video_action_pairs_COIN[video]]

    for action_pair in action_pairs:
        data_train.add(action_pair[0])
        data_train.add(action_pair[1])

    with open("data/action_localization/youcookii_annotations_trainval.json") as f:
        dict_coin = json.loads(f.read())

    for video in dict_coin['database'].keys():
        # annotations_video = dict_coin['database'][video]["annotation"]
        annotations_video = dict_coin['database'][video]["annotations"]
        data_type = dict_coin['database'][video]["subset"]
        # if data_type == "testing":#coin
        if data_type == "validation": #youcook
            for annotation in annotations_video:
                data_test.add(annotation['sentence']) #youcook
        # elif data_type == "training":
        #     for annotation in annotations_video:
        #         data_train.add(annotation['sentence'])  # youcook
    print(len(data_train), len(data_test))
    return list(data_train), list(data_test)

def map_actions_to_graph_actions(dataset):
    print("Mapping actions from validation/test to actions from train graph ...")
    actions_train, actions_test = read_data_train_test(dataset)
    list_stsbrt_embeddings_train = get_sentence_embedding_features(actions_train)
    list_stsbrt_embeddings_test = get_sentence_embedding_features(actions_test)
    sim_matrix = cosine_similarity(list_stsbrt_embeddings_test.cpu(), list_stsbrt_embeddings_train.cpu())
    most_sim_idx = np.argmax(sim_matrix, axis=1)
    # print(len(most_sim_idx))
    # print(len(actions_train), len(actions_test))
    most_sim_actions_test = [actions_train[idx] for idx in most_sim_idx]
    actions_test2train = {}
    for action_test, most_sim_action_train in zip(actions_test, most_sim_actions_test):
        actions_test2train[action_test] = most_sim_action_train
    # for action_test, sim_action_test in zip(actions_test, most_sim_actions_test):
    #     print(action_test, sim_action_test)
    return actions_test2train


def get_sentence_embedding_features(actions):
    # models: https://www.sbert.net/docs/predeved_models.html#semantic-textual-similarity
    model = SentenceTransformer('stsb-roberta-base')
    list_stsbrt_embeddings = model.encode(list(actions), show_progress_bar=True, convert_to_tensor=True)
    return list_stsbrt_embeddings


def get_clip_features(actions, input_file):
    list_features_dict = torch.load(input_file)
    actions_features_dict = {}
    for action in track(actions, description="Getting CLIP txt features for all actions"):
        if action not in actions_features_dict:
            actions_features_dict[action] = {'text': [], 'visual': []}
        for features_dict in list_features_dict:
            action_in_dict = features_dict['action']
            if action_in_dict == action:
                actions_features_dict[action]['text'].append(torch.unsqueeze(features_dict['text_features'], 0))
                actions_features_dict[action]['visual'].append(torch.unsqueeze(features_dict['visual_features'], 0))

    count_actions_not_found = 0
    count_actions_found = 0
    list_txt_clip_embeddings, list_vis_clip_embeddings, actions_processed = [], [], []
    for action in actions_features_dict:
        if not actions_features_dict[action]['text'] or not actions_features_dict[action]['visual']:
            print(f"Action {action} doesn't have CLIP features")
            count_actions_not_found += 1
        else:
            count_actions_found += 1
            actions_processed.append(action)
            mean_txt = torch.mean(torch.cat(actions_features_dict[action]['text'], dim=0), dim=0)
            mean_vis = torch.mean(torch.cat(actions_features_dict[action]['visual'], dim=0), dim=0)
            list_txt_clip_embeddings.append(mean_txt)
            list_vis_clip_embeddings.append(mean_vis)

    console.print(
        f"There are {count_actions_found} actions with CLIP feat, {count_actions_not_found} actions with NO CLIP feat, from {len(actions)} total.",
        style="magenta")
    return list_txt_clip_embeddings, list_vis_clip_embeddings, actions_processed


def get_average_clip_embeddings(input1, input2, output):
    txtclip_node_data = pd.read_csv(input1, index_col=0)
    visclip_node_data = pd.read_csv(input2, index_col=0)

    txtclip_embeddings = txtclip_node_data.values
    visclip_embeddings = visclip_node_data.values
    avg_embeddings = (txtclip_embeddings + visclip_embeddings) / 2
    actions_txt = txtclip_node_data.index.tolist()
    actions_clip = visclip_node_data.index.tolist()
    assert actions_txt == actions_clip

    # Save graph node features: Sentence Bert, Text CLIP, Visual CLIP
    df = pd.DataFrame(avg_embeddings, index=actions_txt)
    df.to_csv(output)


def save_nodes_null_df(action_pairs):
    action_pairs = [ast.literal_eval(action_pair) for action_pair in action_pairs]
    actions = set()
    for action_pair in action_pairs:
        actions.add(action_pair[0])
        actions.add(action_pair[1])

    list_null_embeddings = [torch.zeros(700)] * len(actions)
    df = pd.DataFrame([tensor.cpu().numpy() for tensor in list_null_embeddings], index=actions)
    df.to_csv('data/graph/null_nodes.csv')


def save_nodes_edges_df(action_pairs, transcripts_per_action, TIME_DIFF):
    action_pairs = [ast.literal_eval(action_pair) for action_pair in action_pairs]
    actions = set()
    for action_pair in action_pairs:
        actions.add(action_pair[0])
        actions.add(action_pair[1])

    actions = sorted(actions)
    transcripts = [transcripts_per_action[action] for action in actions]

    list_stsbrt_transcript_embeddings = get_sentence_embedding_features(transcripts)
    list_stsbrt_embeddings = get_sentence_embedding_features(actions)
    list_txt_clip_embeddings, list_vis_clip_embeddings, actions_CLIP = get_clip_features(actions,
                                                                                         input_file='data/clip_features.pt')

    # Save graph nodes
    df = pd.DataFrame([tensor.cpu().numpy() for tensor in list_stsbrt_transcript_embeddings], index=actions)
    df.to_csv('data/graph/txt_transcript_nodes.csv')
    df = pd.DataFrame([tensor.cpu().numpy() for tensor in list_stsbrt_embeddings], index=actions)
    df.to_csv('data/graph/txt_action_nodes.csv')
    df = pd.DataFrame([tensor.cpu().numpy() for tensor in list_txt_clip_embeddings], index=actions_CLIP)
    df.to_csv('data/graph/vis_action_nodes.csv')
    df = pd.DataFrame([tensor.cpu().numpy() for tensor in list_vis_clip_embeddings], index=actions_CLIP)
    df.to_csv('data/graph/vis_video_nodes.csv')
    get_average_clip_embeddings(input1='data/graph/vis_action_nodes.csv',
                                input2='data/graph/vis_video_nodes.csv',
                                output='data/graph/vis_action_video_nodes.csv')

    # Save graph edges
    counter = Counter(tuple(x) for x in action_pairs)
    actions_not_found = list(set(actions) - set(actions_CLIP))
    print(f"Actions not found with CLIP (missing videos): {actions_not_found}")
    list_tuples_actions = [(action_pair[0], action_pair[1], counter[action_pair]) for action_pair in counter
                           if action_pair[0] not in actions_not_found and action_pair[1] not in actions_not_found]

    df = pd.DataFrame(list_tuples_actions, columns=['source', 'target', 'weight'])
    # df.to_csv(f'data/graph/edges_{TIME_DIFF}.csv', index=False)
    df.to_csv(f'data/graph/edges.csv', index=False)


def convert_datetime_to_seconds(clip_a1):
    clip_a = [clip_a1[0].split(".")[0], clip_a1[1].split(".")[0]]
    date_time = datetime.datetime.strptime(clip_a[1], '%H:%M:%S')
    a_timedelta = date_time - datetime.datetime(1900, 1, 1)
    seconds = a_timedelta.total_seconds()
    return seconds


def filter_actions():
    with open('data/dict_video_actions.json') as json_file:
        dict_verb_dobj_per_video = json.load(json_file)

    set_verbs = set()
    set_actions = set()
    for values in dict_verb_dobj_per_video.values():
        for v in values:
            action = ast.literal_eval(v)[0]
            verb = action.split()[0]
            set_verbs.add(verb)
            set_actions.add(action)

    filter_out_actions = [action for action in set_actions if
                          len(action.split()) == 2 and nltk.pos_tag(action.split())[1][1] in ['IN']]
    rename_actions = {}
    for verb in track(set_verbs, description="Renaming actions.."):
        actions_with_verb = [action for action in set_actions if action.split()[0] == verb]
        for action_i in actions_with_verb:
            for action_j in actions_with_verb:
                if action_i != action_j:
                    words_action_i = action_i.split()
                    words_action_j = action_j.split()
                    if set(words_action_j).issubset(set(words_action_i)) and action_j not in filter_out_actions:
                        if action_i not in rename_actions:
                            rename_actions[action_i] = []
                        rename_actions[action_i].append(action_j)
            if action_i in rename_actions:
                rename_actions[action_i] = min(rename_actions[action_i], key=lambda x: len(x))

    dict_renamed_actions = {}
    new_actions = set()
    for key, values in dict_verb_dobj_per_video.items():
        new_values = []
        for v in values:
            [action, sentence, [time_start, time_end]] = ast.literal_eval(v)
            if action in rename_actions:
                action = rename_actions[action]
            new_actions.add(action)
            new_values.append(str([action, sentence, [time_start, time_end]]))
        dict_renamed_actions[key] = new_values

    console.print(f"#Unique actions initial, : {len(new_actions)}", style="magenta")

    with open('data/dict_renamed_actions.json', 'w+') as fp:
        json.dump(dict_renamed_actions, fp)


def main():
    video_sample = "hK7yV276110"
    TIME_DIFF = 10  # 5, 10, 15

    ''' Getting all action pairs'''
    verbs = get_action_verbs()
    get_actions(verbs, video_sample, try_per_video=False)  # saves the data
    filter_actions()  # filter out actions(say, talk, ..) and rename actions that are included in each other (same verb and Dobj)

    get_action_pairs_by_time(TIME_DIFF)  # get stats initial

    plot_graph_actions(video_sample, input="data/dict_video_action_pairs.json")
    get_stats_actions(before_clustering=True)

    ''' Custering all actions '''
    cluster_actions()  # saves the data
    filtered_clusters = filter_clusters_by_size()
    filter_pairs_by_cluster(filtered_clusters, TIME_DIFF)  # saves the data

    plot_graph_actions(video_sample, input="data/dict_video_action_pairs_filtered_by_cluster.json")
    get_stats_actions(before_clustering=False)

    ''' Combining the graph_plots for all videos '''
    action_pairs, transcripts_per_action = combine_graphs(video_sample, TIME_DIFF, directed=False, sample=False,
                                                          filter_by_link=True)

    # show_graph_actions(action_pairs, video="all_videos")
    # ''' Saving dataframes for all nodes and edges '''
    save_nodes_edges_df(action_pairs, transcripts_per_action, TIME_DIFF)
    # save_nodes_null_df(action_pairs)


if __name__ == '__main__':
    main()

import ast
import datetime
import json
from collections import Counter
import numpy as np
import pandas as pd
import spacy
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from pyvis.network import Network

nlp = spacy.load("en_core_web_sm")  # TODO: try trf


def get_all_action_verbs():
    print("Getting all the verbs ...")
    # http://www-personal.umich.edu/~jlawler/levin.verbs
    with open('data/utils/levin_verbs.txt') as f:  # TODO: MAYBE add more from file
        levin_verbs = f.readlines()
    levin_verbs = levin_verbs[0].split()
    # print(levin_verbs)

    with open('data/utils/visual_verbnet_beta2015.json') as json_file:
        visual_verbnet_beta2015 = json.load(json_file)

    list_actions = []
    for data in visual_verbnet_beta2015["visual_actions"]:
        if data["category"] not in ["communication", "perception", "posture"]:
            action = data["name"].split("_")[0]
            if action not in ["follow", "be", "show", "reach", "laugh", "avoid", "get", "feed", "run", "carry", "give",
                              "make"] and action not in list_actions:
                list_actions.append(action)
    # print(len(list_actions))

    # combine levin & VisualVerbNet verbs
    all_actions = list(set(list_actions + levin_verbs))
    # print(len(all_actions))

    return all_actions
    # for x in []:
    #     all_actions.remove(x)

    # # combine levin & VisualVerbNet & Synonim & Hyponim verbs
    # from nltk.corpus import wordnet
    # from itertools import chain
    #
    # for verb in tqdm(all_actions):
    #     # print(verb)
    #     ss = wordnet.synset(verb + ".v.01")
    #     hyponims = list(set([w for s in ss.closure(lambda s: s.hyponyms()) for w in s.lemma_names()]))
    #
    #     all_actions += hyponims
    #
    #     synonyms = wordnet.synsets(verb)
    #     synonyms = list(set(chain.from_iterable([word.lemma_names() for word in synonyms if word._pos == 'v'])))
    #     all_actions += synonyms
    #
    # all_actions = list(set(all_actions))
    # print(all_actions)
    # print(len(all_actions))


def get_VP(t, action):
    for tok in t.children:
        if tok.dep_ in ["dobj", "prep", "pobj", "compound"]:
            if tok.pos_ in ['ADP', 'NOUN', 'PROPN']:
                action = action + [tok]
            action = get_VP(tok, action)
    return action


def get_all_verb_dobj(all_coref_sentences, verbs):
    remove_words = ["some", "mine", "ooh", "thing", "lot", "whatever", "whichever", "much", "many",
                    "most", "bunch", "all", "one", "do", "one", "kind", "it", "a", "thumbs", "bit", "what", "which",
                    "who", "whom", "I", "him", "her", "something", "anything", "element",
                    "that", "those", "them", "this", "these", "sense", "be", "decision", "subscribe", "like", "love",
                    "comment", "they", "link", "same", "different", "everything", "nothing", "button", "video"]

    list_all_actions = []
    for doc in nlp.pipe(all_coref_sentences):
        list_actions = []
        for t in doc:
            if t.pos_ == "VERB" and t.lemma_ in verbs:
                action = get_VP(t, [t])
                sorted_all = [(tok.lemma_, tok.pos_) for tok in sorted(action, key=lambda tok: tok.i) if
                              tok.lemma_ not in remove_words and tok.lemma_.isalpha()]

                # remove everything before verb
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
                 x not in sorted_words_no_duplicates]  # and not x[0].isupper()
                if len(sorted_words_no_duplicates) >= 2:
                    # sorted_words_no_duplicates = sorted_words_no_duplicates[:5]
                    list_actions.append(" ".join(sorted_words_no_duplicates))
                    # if "take egg with" in list_actions:
                    #     print(doc.text)
        list_all_actions.append(list_actions)
    #     if not list_actions:
    #         list_all_actions.append(doc.text)
    #     else:
    #         list_all_actions.append(list_actions)
    # for actions in list_all_actions:
    #     print(actions)
    # print(list_all_actions)
    return list_all_actions


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
    return difference


def show_graph_actions(clustered_action_pairs, video):
    counter = Counter([tuple(x) for x in clustered_action_pairs])
    list_tuples_actions = [(action_pair[0], action_pair[1], counter[action_pair]) for action_pair in counter]
    df = pd.DataFrame(list_tuples_actions, columns=['action1', 'action2', 'nb_examples'])
    sources = df['action1']
    targets = df['action2']
    weights = df['nb_examples']

    net = Network(height='100%', width='100%', bgcolor='#222222', font_color='white')

    # set the physics layout of the network
    # net.barnes_hut()
    net.hrepulsion()

    # edge_data = zip(sources, targets)
    for index, (src, dst, w) in enumerate(zip(sources, targets, weights)):
        # color differently start and end nodes:
        # if index == 0:
        #     net.add_node(src, src, title=src, color='red', value=100)
        #     net.add_node(dst, dst, title=dst, value=100)
        #     net.add_edge(src, dst, value=w)
        # elif index == len(sources) - 1:
        #     net.add_node(src, src, title=src, value=100)
        #     net.add_node(dst, dst, title=dst, color='yellow', value=100)
        #     net.add_edge(src, dst, value=w)
        # else:
        #     net.add_node(src, src, title=src, value=100)
        #     net.add_node(dst, dst, title=dst, value=100)
        #     # net.add_edge(src, dst, width=1)
        #     net.add_edge(src, dst, value=w)
        # print(index, src + " : " + dst)
        net.add_node(src, label=src, title=str(index), value=100)
        net.add_node(dst, label=dst, title=str(index), value=100)
        # net.add_edge(src, dst, width=1)
        net.add_edge(src, dst, label=str(index), title=str(index) + ": " + src + " , " + dst, value=w)

    net.show_buttons(filter_=['physics'])
    net.show("data/graphs/" + video + '.html')


def get_all_action_pairs(all_verbs, video_sample, try_per_video):
    with open('data/coref_all_sentence_transcripts.json') as json_file:
        # with open('data/UCM3P_G21gOSVdepXrEFojIg.json') as json_file:
        all_sentence_transcripts_rachel = json.load(json_file)

    dict_verb_dobj_per_video = {}
    wordsToBeRemoved = ["some", "few", "bit", "slice", "pinch", "little", "lot", "much", "many", "more", "about"]

    print("Get all actions from {0} videos".format(len(all_sentence_transcripts_rachel)))
    for video in tqdm(all_sentence_transcripts_rachel):
        if try_per_video and video != video_sample:
            continue
        actions_per_video = []
        all_coref_sentences, all_time_start, all_time_end, all_sentences = [], [], [], []
        for dict in all_sentence_transcripts_rachel[video]:
            sentence, coref_sentence, time_s, time_e, list_mentions = dict["sentence"], dict["coref_sentence"], \
                                                                      dict["time_s"], dict["time_e"], dict[
                                                                          "list_mentions"]

            coref_sentence = " ".join([value for value in coref_sentence.split() if value not in wordsToBeRemoved])

            all_coref_sentences.append(coref_sentence)
            all_time_start.append(str(datetime.timedelta(seconds=time_s)))
            all_time_end.append(str(datetime.timedelta(seconds=time_e)))
            all_sentences.append(sentence)

        all_actions_per_video = get_all_verb_dobj(all_coref_sentences, all_verbs)
        # print(all_actions_per_video)
        for (all_actions_per_sentence, time_start, time_end, sentence) in zip(all_actions_per_video, all_time_start,
                                                                              all_time_end, all_sentences):
            if all_actions_per_sentence:
                actions_per_video += [str([action, sentence, [time_start, time_end]]) for action in
                                      all_actions_per_sentence]
        dict_verb_dobj_per_video[video] = actions_per_video

    with open('data/dict_video_action_pairs.json', 'w+') as fp:
        json.dump(dict_verb_dobj_per_video, fp)


def filter_action_pairs_by_time():
    DIFF_TIME = 10  # seconds
    print(f"Filtering action pairs by time: {DIFF_TIME} seconds ...")

    with open('data/dict_video_action_pairs.json') as json_file:
        dict_verb_dobj_per_video = json.load(json_file)

    dict_video_action_pairs = {}
    for video in tqdm(dict_verb_dobj_per_video):
        dict_video_action_pairs[video] = []
        all_actions_time = [ast.literal_eval(action_data) for action_data in dict_verb_dobj_per_video[video]]
        for i in range(0, len(all_actions_time) - 1):
            for j in range(i + 1, len(all_actions_time)):
                difference = get_diff_time(all_actions_time[i][2], all_actions_time[j][2])
                if difference > DIFF_TIME:
                    break
                action_1, transcript_a1, clip_a1 = all_actions_time[i]
                action_2, transcript_a2, clip_a2 = all_actions_time[j]
                if action_1 != action_2 and [(action_1, transcript_a1, clip_a1),
                                             (action_2, transcript_a2, clip_a2)] not in dict_video_action_pairs[video] \
                        and [(action_2, transcript_a2, clip_a2), (action_1, transcript_a1, clip_a1)] not in \
                        dict_video_action_pairs[video]:
                    dict_video_action_pairs[video].append(
                        [(action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)])
                    # dict_video_action_pairs[video].append([(action_1, clip_a1), (action_2, clip_a2), difference])
                    # print((action_1, clip_a1), (action_2, clip_a2), difference)
                    # print((action_1, transcript_a1), (action_2, transcript_a2))

    return dict_video_action_pairs


def plot_graph_actions(dict_video_action_pairs, video):
    action_pairs = [sorted((action_1, action_2))
                    for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
                    in dict_video_action_pairs[video]]
    show_graph_actions(action_pairs, video)


def get_stats_actions(dict_video_action_pairs, before_clustering):
    all_actions = []
    all_action_pairs = []
    for video in dict_video_action_pairs:
        for [(action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)] in dict_video_action_pairs[video]:
            all_actions.append(action_1)
            all_actions.append(action_2)
            all_action_pairs.append((action_1, action_2))

    dict_per_verb = {}
    for s in list(set(all_actions)):
        verb = s.split()[0]
        if s.split()[0] not in dict_per_verb:
            dict_per_verb[verb] = []
        dict_per_verb[verb].append(s)

    word = "before" if before_clustering else "after"
    print(f"# actions after filtering, {word} clustering: {len(all_actions)}")
    print(f"# action pairs after filtering,  {word} clustering: {len(all_action_pairs)}")
    print("--------------------------------------------------------------")
    print(f"# unique actions after filtering,  {word} clustering: {len(set(all_actions))}")
    print(f"# unique action pairs after filtering,  {word} clustering: {len(set(all_action_pairs))}")

    # print(Counter(all_actions).most_common(100))
    # print("-----")
    dict_per_verb = {k: v for k, v in sorted(dict_per_verb.items(), key=lambda item: len(item[1]), reverse=True)}
    # for key in dict_per_verb:
    #     print(key, len(dict_per_verb[key]))
    print(f"# unique verbs: {len(dict_per_verb)}")
    # print(Counter(all_action_pairs).most_common(10))


def cluster_actions(dict_video_action_pairs):
    set_actions = set()
    for video in dict_video_action_pairs:
        for [(action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)] in dict_video_action_pairs[video]:
            set_actions.add(action_1)
            set_actions.add(action_2)

    list_actions = list(set_actions)
    print(f"Clustering {len(list_actions)} actions ...")
    # list_actions = list_actions[:1000]
    model = SentenceTransformer(
        'stsb-roberta-base')  # models: https://www.sbert.net/docs/predeved_models.html#semantic-textual-similarity
    list_embeddings = model.encode(list_actions, show_progress_bar=True, convert_to_tensor=True)

    print("Start clustering")

    # Normalize the embeddings to unit length
    corpus_embeddings = list_embeddings / np.linalg.norm(list_embeddings, axis=1, keepdims=True)

    # Perform agglomerative clustering
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
        # print("Cluster ", i + 1)
        # print(cluster)
        # print("")
        dict_clustered_actions[cluster[0]] = cluster[1:]  # TODO? Rename clusters

    with open('data/dict_clustered_actions.json', 'w+') as fp:
        json.dump(dict_clustered_actions, fp)
    return dict_clustered_actions


def filter_clusters_by_size():
    CLUSTER_SIZE = 1
    print(f"Filtering clusters by size {CLUSTER_SIZE} ...")
    with open('data/dict_clustered_actions.json') as json_file:
        dict_clustered_actions = json.load(json_file)

    filtered_clusters = {}
    count = 0
    for key in dict_clustered_actions:
        if len(dict_clustered_actions[key]) < CLUSTER_SIZE:
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
    # for key, value in my_dict.items():
    #     if val in value:
    #         return key
    # return None


def filter_pairs_by_cluster(dict_video_action_pairs, filtered_clusters):
    print("Filtering action pairs by clusters ...")
    dict_video_action_pairs_filtered = {}
    for video in tqdm(dict_video_action_pairs):
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

    with open('data/dict_video_action_pairs_filtered.json', 'w+') as fp:
        json.dump(dict_video_action_pairs_filtered, fp)
    # return dict_video_action_pairs_filtered


def combine_graphs(sample, filter_by_link):
    with open('data/dict_video_action_pairs_filtered.json') as json_file:
        dict_video_action_pairs_filtered = json.load(json_file)
    if sample:
        SAMPLE_SIZE = 5
        all_action_pairs = [str(sorted((action_1, action_2)))
                            for video in list(dict_video_action_pairs_filtered)[:SAMPLE_SIZE]
                            for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
                            in dict_video_action_pairs_filtered[video]]
    else:
        all_action_pairs = [str(sorted((action_1, action_2)))
                            for video in dict_video_action_pairs_filtered
                            for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
                            in dict_video_action_pairs_filtered[video]]

    print(f"Having {len(all_action_pairs)} action pairs and {len(set(all_action_pairs))} unique ones")
    if filter_by_link:
        counter = Counter(all_action_pairs)
        print("Removing links that appear only once ...")
        all_action_pairs = [action_pair for action_pair in all_action_pairs if counter[action_pair] > 1]
        print(
            f"After filtering, having {len(all_action_pairs)} action pairs and {len(set(all_action_pairs))} unique ones")

    all_action_pairs = [ast.literal_eval(action_pair) for action_pair in all_action_pairs]
    # # print(Counter(all_action_pairs).most_common(50))

    # nb_pairs_appear_1 = sum(1 for count in counter.values() if count == 1)
    # print(f"Having {nb_pairs_appear_1} action pairs appear 1 time")

    # sorted_counter_values = sorted(counter.values(), reverse=True)
    # print(Counter(sorted_counter_values).most_common())
    return all_action_pairs


def main():
    all_verbs = get_all_action_verbs()
    # video_sample = "hK7yV276110"
    # video_sample = "SyMHOV6HhyI"
    # video_sample = "zXqBCqPa9VY"
    video_sample = "34uV2sJlF9Y"

    # get_all_action_pairs(all_verbs, video_sample, try_per_video=False)  # saves the data
    # dict_video_action_pairs = filter_action_pairs_by_time()

    # plot_graph_actions(dict_video_action_pairs, video_sample)
    # get_stats_actions(dict_video_action_pairs, before_clustering=True)

    # dict_clustered_actions = cluster_actions(dict_video_action_pairs)
    # filtered_clusters = filter_clusters_by_size()
    # filter_pairs_by_cluster(dict_video_action_pairs, filtered_clusters)

    # plot_graph_actions(filtered_dict_video_action_pairs, video_sample)
    # get_stats_actions(filtered_dict_video_action_pairs, before_clustering=False)

    all_action_pairs = combine_graphs(sample=True, filter_by_link=True)
    show_graph_actions(all_action_pairs, video="all_videos")


if __name__ == '__main__':
    main()

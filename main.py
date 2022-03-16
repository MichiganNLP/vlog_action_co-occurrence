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
from pyvis.network import Network
from rich.progress import track
from rich.console import Console
from rich.pretty import pprint

console = Console()

# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_trf")  # TODO: try trf


def get_all_action_verbs():
    print("Getting all the verbs ...")
    # http://www-personal.umich.edu/~jlawler/levin.verbs
    with open('data/utils/dict_levin_verbs.json') as json_file:
        levin_verbs_dict = json.load(json_file)
    levin_verbs = list(set([v for values in levin_verbs_dict.values() for v in values.split()]))

    with open('data/utils/visual_verbnet_beta2015.json') as json_file:
        visual_verbnet_beta2015 = json.load(json_file)
    list_verbnet_actions = [data["name"].split("_")[0] for data in visual_verbnet_beta2015["visual_actions"]
                            if data["category"] not in ["communication", "perception", "posture"]]

    verbs_to_remove = ["follow", "like", "subscribe", "be", "show", "reach", "laugh", "avoid", "get", "feed",
                       "run", "carry", "give", "make", "show", "give", "click", "turn", "recommend", "link",
                       "come", "donate", "help", "hang", "hit", "stream", "kick", "blink", "donate", "bump", "play"]
    levin_verbs = [v for v in levin_verbs if v not in verbs_to_remove]
    list_verbnet_actions = [v for v in list_verbnet_actions if v not in verbs_to_remove]
    # list_extra_verbs = ["moisturize", "disinfect", "scrub", "declutter", "exfoliate"]
    # combine Levin & VisualVerbNet verbs
    console.print(f"There are {len(levin_verbs)} Levin and {len(list_verbnet_actions)} VisualVerbNet verbs.",
                  style="magenta")
    all_actions = list(set(list_verbnet_actions + levin_verbs))
    console.print(f"There are {len(all_actions)} verbs in total.", style="magenta")

    return all_actions
    # for x in []:
    #     all_actions.remove(x)

    # # combine levin & VisualVerbNet & Synonim & Hyponim verbs
    # from nltk.corpus import wordnet
    # from itertools import chain
    #
    # for verb in track(all_actions):
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
                    "most", "bunch", "all", "one", "do", "one", "kind", "it", "a", "thumbs", "what", "which",
                    "who", "whom", "I", "him", "her", "something", "anything", "element",
                    "that", "those", "them", "this", "these", "sense", "be", "decision", "subscribe", "like", "love",
                    "comment", "they", "link", "same", "different", "everything", "nothing", "button", "video",
                    "button"]

    list_all_actions = []
    list_not_caught_verbs = []
    for doc in nlp.pipe(all_coref_sentences):
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
                    # sorted_words_no_duplicates = sorted_words_no_duplicates[:5]
                    list_actions.append(" ".join(sorted_words_no_duplicates))
                    # if "add in" in list_actions:
                    #     print(doc.text)
        list_all_actions.append(list_actions)
    #     if not list_actions:
    #         list_all_actions.append(doc.text)
    #     else:
    #         list_all_actions.append(list_actions)
    # for actions in list_all_actions:
    #     print(actions)
    # print(list_all_actions)
    return list_all_actions, list_not_caught_verbs


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
    # clustered_action_pairs = [ast.literal_eval(action_pair) for action_pair in clustered_action_pairs]
    clustered_action_pairs = [action_pair for action_pair in clustered_action_pairs]

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
    net.show("data/graph_plots/" + video + '.html')


def get_all_actions(all_verbs, video_sample, try_per_video):
    with open('data/coref_all_sentence_transcripts.json') as json_file:
        all_sentence_transcripts_rachel = json.load(json_file)

    dict_verb_dobj_per_video = {}
    wordsToBeRemoved = ["your", "some", "few", "bit of", "slice", "pinch", "little", "lot", "much", "many", "more",
                        "about"]

    all_not_caught_verbs = []
    for video in track(all_sentence_transcripts_rachel, description="Getting all actions..."):
        if try_per_video and video != video_sample:
            continue
        actions_per_video = []
        all_coref_sentences, all_time_start, all_time_end, all_sentences = [], [], [], []
        for dict in all_sentence_transcripts_rachel[video]:
            sentence, coref_sentence, time_s, time_e, list_mentions = dict["sentence"], dict["coref_sentence"], \
                                                                      dict["time_s"], dict["time_e"], dict[
                                                                          "list_mentions"]

            # coref_sentence = " ".join([value for value in coref_sentence.split() if value not in wordsToBeRemoved])
            for w in wordsToBeRemoved:
                coref_sentence = coref_sentence.replace(" " + w + " ", " ")

            all_coref_sentences.append(coref_sentence)
            all_time_start.append(str(datetime.timedelta(seconds=time_s)))
            all_time_end.append(str(datetime.timedelta(seconds=time_e)))
            all_sentences.append(sentence)

        all_actions_per_video, list_not_caught_verbs = get_all_verb_dobj(all_coref_sentences, all_verbs)
        # for v in list_not_caught_verbs:
        #     all_not_caught_verbs.append(v)

        for (all_actions_per_sentence, time_start, time_end, sentence) in zip(all_actions_per_video, all_time_start,
                                                                              all_time_end, all_sentences):
            if all_actions_per_sentence:
                actions_per_video += [str([action, sentence, [time_start, time_end]]) for action in
                                      all_actions_per_sentence]
        dict_verb_dobj_per_video[video] = actions_per_video

    # pprint(Counter(all_not_caught_verbs).most_common(550))
    # all_not_caught_verbs = list(set(all_not_caught_verbs))
    # print(f"There are {len(all_not_caught_verbs)} not caught verbs ..")
    with open('data/dict_video_actions.json', 'w+') as fp:
        json.dump(dict_verb_dobj_per_video, fp)


def get_action_pairs_by_time():
    time_difference = 10  # seconds
    with open('data/dict_video_actions.json') as json_file:
        dict_verb_dobj_per_video = json.load(json_file)

    dict_video_action_pairs = {}
    all_actions = []
    for video in track(dict_verb_dobj_per_video, description=f"Getting action pairs by time diff {time_difference}..."):
        dict_video_action_pairs[video] = []
        all_actions_time = [ast.literal_eval(action_data) for action_data in dict_verb_dobj_per_video[video]]
        for i in range(0, len(all_actions_time) - 1):
            for j in range(i + 1, len(all_actions_time)):
                all_actions.append(all_actions_time[i][0])
                difference = get_diff_time(all_actions_time[i][2], all_actions_time[j][2])
                if difference > time_difference:
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
    console.print(f"#Unique actions before time filtering: {len(set(all_actions))}", style="magenta")
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

    all_verbs, all_actions, all_action_pairs = [], [], []
    for video in dict_video_action_pairs:
        for [(action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)] in dict_video_action_pairs[video]:
            all_actions.append(action_1)
            all_actions.append(action_2)
            all_action_pairs.append(str(sorted((action_1, action_2))))
            all_verbs.append(action_1.split()[0])
            all_verbs.append(action_2.split()[0])

    word = "before" if before_clustering else "after"
    console.print(f"#Unique actions after time filtering, {word} clustering: {len(set(all_actions))}", style="magenta")
    console.print(f"#Unique action pairs after time filtering, {word} clustering: {len(set(all_action_pairs))}",
                  style="magenta")
    console.print(f"#unique verbs after time filtering, {word} clustering: {len(set(all_verbs))}", style="magenta")
    # pprint(f"20 most common verbs: {Counter(all_verbs).most_common(20)}", expand_all=True)
    # pprint(f"20 most common actions: {Counter(all_actions).most_common(20)}", expand_all=True)
    # pprint(f"20 most common action pairs: {Counter(all_action_pairs).most_common(20)}", expand_all=True)
    return set(all_verbs)


def cluster_actions():
    with open('data/dict_video_action_pairs.json') as json_file:
        dict_video_action_pairs = json.load(json_file)
    all_actions = []
    for video in dict_video_action_pairs:
        for [(action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)] in dict_video_action_pairs[video]:
            all_actions.append(action_1)
            all_actions.append(action_2)

    list_actions = list(set(all_actions))
    print(f"Clustering {len(list_actions)} unique actions ...")
    # list_actions = list_actions[:1000]
    model = SentenceTransformer(
        'stsb-roberta-base')  # models: https://www.sbert.net/docs/predeved_models.html#semantic-textual-similarity
    list_embeddings = model.encode(list_actions, show_progress_bar=True, convert_to_tensor=True)

    print("Start clustering")

    # Normalize the embeddings to unit length
    corpus_embeddings = list_embeddings / np.linalg.norm(list_embeddings, axis=1, keepdims=True)

    # Perform agglomerative clustering
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=1.5)  # TODO: might need to change threshold
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
        clustered_sentences[cluster_id].append(list_actions[sentence_id])

    # assign name of cluster - the most common action from cluster
    dict_clustered_actions = {}
    occurrences = Counter(all_actions)
    for i, cluster in clustered_sentences.items():
        # dict_clustered_actions[cluster[0]] = cluster[1:]
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


def clean_clusters():
    # by verb, by object?, merge clusters that have same verb
    with open('data/dict_clustered_actions.json') as json_file:
        dict_clustered_actions = json.load(json_file)
    clean_clustered_actions = {}
    for action_cluster_name, values in dict_clustered_actions.items():
        if action_cluster_name not in clean_clustered_actions:
            clean_clustered_actions[action_cluster_name] = []
        cluster_name_words = action_cluster_name.split()
        for action in values:
            action_words = action.split()
            action_in_cluster = False
            for w in action_words:
                if w in cluster_name_words:
                    action_in_cluster = True
                    break
            if action_in_cluster:
                clean_clustered_actions[action_cluster_name].append(action)
            else:
                for new_action_cluster_name in dict_clustered_actions:
                    if action.split()[0] == new_action_cluster_name.split()[0]:
                        action_in_cluster = True
                        if new_action_cluster_name not in clean_clustered_actions:
                            clean_clustered_actions[new_action_cluster_name] = []
                        clean_clustered_actions[new_action_cluster_name].append(action)
                        console.print(f"{action}: {action_cluster_name} -> {new_action_cluster_name} ", style="magenta")
                        break
                # if not action_in_cluster:




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
    # for key, value in my_dict.items():
    #     if val in value:
    #         return key
    # return None


def filter_pairs_by_cluster(filtered_clusters):
    with open('data/dict_video_action_pairs.json') as json_file:
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

    with open('data/dict_video_action_pairs_filtered_by_cluster.json', 'w+') as fp:
        json.dump(dict_video_action_pairs_filtered, fp)


def combine_graphs(video_sample, sample, filter_by_link):
    with open('data/dict_video_action_pairs_filtered_by_cluster.json') as json_file:
        dict_video_action_pairs_filtered = json.load(json_file)
    if sample:
        # SAMPLE_SIZE = 20
        # all_action_pairs = [str(sorted((action_1, action_2)))
        #                     for video in list(dict_video_action_pairs_filtered)[:SAMPLE_SIZE]
        #                     for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
        #                     in dict_video_action_pairs_filtered[video]]

        all_action_pairs = [str(sorted((action_1, action_2)))
                            for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
                            in dict_video_action_pairs_filtered[video_sample]]
    else:
        all_action_pairs = [str(sorted((action_1, action_2)))
                            for video in dict_video_action_pairs_filtered
                            for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
                            in dict_video_action_pairs_filtered[video]]

    if filter_by_link:
        counter = Counter(all_action_pairs)
        print("Removing links that appear only once ...")
        all_action_pairs = [action_pair for action_pair in all_action_pairs if counter[action_pair] > 1]
        console.print(
            f"After edge filtering, having {len(all_action_pairs)} action pairs and {len(set(all_action_pairs))} unique ones",
            style="magenta")
        dict_video_action_pairs_filtered_by_link = {}
        for video in dict_video_action_pairs_filtered:
            for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2) in dict_video_action_pairs_filtered[video]:
                if str(sorted((action_1, action_2))) in all_action_pairs:
                    if video not in dict_video_action_pairs_filtered_by_link:
                        dict_video_action_pairs_filtered_by_link[video] = []
                    dict_video_action_pairs_filtered_by_link[video].append([(action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)])

        with open('data/dict_video_action_pairs_filtered_by_link.json', 'w+') as fp:
            json.dump(dict_video_action_pairs_filtered_by_link, fp)

    all_action_pairs_converted = [ast.literal_eval(action_pair) for action_pair in all_action_pairs]
    all_actions = []
    verbs = []
    for (a1, a2) in all_action_pairs_converted:
        all_actions.append(a1)
        all_actions.append(a2)
        verbs.append(a1.split()[0])
        verbs.append(a2.split()[0])
    console.print(f"After filtering, having {len(all_actions)} actions and {len(set(all_actions))} unique ones",
                  style="magenta")
    console.print(f"After filtering, having {len(verbs)} verbs and {len(set(verbs))} unique ones", style="magenta")

    for (action_1, action_2) in track(all_action_pairs_converted, description="Checking for repeating pairs..."):
        if (action_2, action_1) in all_action_pairs_converted:
            raise ValueError("Error! Found repeating pair:" + str((action_1, action_2)))

    # # print(Counter(all_action_pairs_converted).most_common(50))

    # nb_pairs_appear_1 = sum(1 for count in counter.values() if count == 1)
    # print(f"Having {nb_pairs_appear_1} action pairs appear 1 time")

    # sorted_counter_values = sorted(counter.values(), reverse=True)
    # print(Counter(sorted_counter_values).most_common())
    return all_action_pairs


def save_nodes_edges_df(all_action_pairs, name):
    all_action_pairs = [ast.literal_eval(action_pair) for action_pair in all_action_pairs]
    all_actions = set()
    for action_pair in all_action_pairs:
        all_actions.add(action_pair[0])
        all_actions.add(action_pair[1])

    model = SentenceTransformer(
        'stsb-roberta-base')  # models: https://www.sbert.net/docs/predeved_models.html#semantic-textual-similarity
    list_stsbrt_embeddings = model.encode(list(all_actions), show_progress_bar=True, convert_to_tensor=True)

    df = pd.DataFrame([tensor.numpy() for tensor in list_stsbrt_embeddings], index=all_actions)
    # TODO: Get CLIP text and video embeddings

    df.to_csv('data/graph/' + name + "_nodes.csv")

    counter = Counter([tuple(x) for x in all_action_pairs])
    list_tuples_actions = [(action_pair[0], action_pair[1], counter[action_pair]) for action_pair in counter]
    df = pd.DataFrame(list_tuples_actions, columns=['source', 'target', 'weight'])
    df.to_csv('data/graph/' + name + "_edges.csv", index=False)


def convert_datetime_to_seconds(clip_a1):
    clip_a = [clip_a1[0].split(".")[0], clip_a1[1].split(".")[0]]
    date_time = datetime.datetime.strptime(clip_a[1], '%H:%M:%S')
    a_timedelta = date_time - datetime.datetime(1900, 1, 1)
    seconds = a_timedelta.total_seconds()
    return seconds


def split_graph_by_time(video_sample, sample=False):
    with open('data/dict_video_action_pairs_filtered_by_cluster.json') as json_file:
        dict_video_action_pairs_filtered = json.load(json_file)

    time_threshold = 5 * 60  # 5 minutes
    if sample:
        # SAMPLE_SIZE = 20
        # all_action_pairs = [str(sorted((action_1, action_2)))
        #                     for video in list(dict_video_action_pairs_filtered)[:SAMPLE_SIZE]
        #                     for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
        #                     in dict_video_action_pairs_filtered[video]]

        all_action_pairs_before = [str(sorted((action_1, action_2)))
                                   for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
                                   in dict_video_action_pairs_filtered[video_sample] if convert_datetime_to_seconds(
                clip_a1) < time_threshold and convert_datetime_to_seconds(clip_a2) < time_threshold]

        all_action_pairs = [str(sorted((action_1, action_2)))
                            for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
                            in dict_video_action_pairs_filtered[video_sample]]
    else:
        all_action_pairs_before = [str(sorted((action_1, action_2)))
                                   for video in dict_video_action_pairs_filtered
                                   for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
                                   in dict_video_action_pairs_filtered[video] if convert_datetime_to_seconds(
                clip_a1) < time_threshold and convert_datetime_to_seconds(clip_a2) < time_threshold]

        all_action_pairs = [str(sorted((action_1, action_2)))
                            for video in dict_video_action_pairs_filtered
                            for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
                            in dict_video_action_pairs_filtered[video]]

    # all_action_pairs_before = [ast.literal_eval(action_pair) for action_pair in all_action_pairs_before]
    # all_action_pairs = [ast.literal_eval(action_pair) for action_pair in all_action_pairs]
    return all_action_pairs_before, all_action_pairs


def main():
    video_sample = "hK7yV276110"
    # video_sample = "SyMHOV6HhyI"
    # video_sample = "zXqBCqPa9VY"
    # video_sample = "amC9EKYmF-s"
    #
    ''' Getting all action pairs'''
    # all_verbs = get_all_action_verbs()
    # get_all_actions(all_verbs, video_sample, try_per_video=False)  # saves the data
    # get_action_pairs_by_time()

    # plot_graph_actions(video_sample, input="data/dict_video_action_pairs.json")
    get_stats_actions(before_clustering=True)

    # ''' Custering all actions '''
    # cluster_actions()  # saves the data
    # # clean_clusters()
    # filtered_clusters = filter_clusters_by_size()
    # filter_pairs_by_cluster(filtered_clusters)  # saves the data

    # plot_graph_actions(video_sample, input="data/dict_video_action_pairs_filtered_by_cluster.json")
    get_stats_actions(before_clustering=False)

    #
    ''' Combining the graph_plots for all videos '''
    all_action_pairs = combine_graphs(video_sample, sample=False, filter_by_link=True)
    # show_graph_actions(all_action_pairs, video="all_videos")
    #
    # # ''' Saving dataframes for all nodes and edges '''
    # save_nodes_edges_df(all_action_pairs, name="all")
    #

    ''' Split graph by time -- might not need this'''
    # all_action_pairs_before, all_action_pairs = split_graph_by_time(video_sample, sample=True)
    # list_actions_not_appearing_before = set(all_action_pairs) - set(all_action_pairs_before)
    # print(f"{len(list_actions_not_appearing_before)} action pairs do not connect before out of {len(list(set(all_action_pairs)))} total action pairs")
    #
    # save_nodes_edges_df(all_action_pairs_before, name="before_" + video_sample)
    # save_nodes_edges_df(all_action_pairs, name="all_" + video_sample)
    # show_graph_actions(all_action_pairs_before, video="before_" + video_sample)
    # show_graph_actions(all_action_pairs_before, video="all_" + video_sample)


if __name__ == '__main__':
    main()

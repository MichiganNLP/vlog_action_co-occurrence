import json
from collections import Counter
import spacy
import time
import ast
import datetime
import en_core_web_sm
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
import numpy as np

nlp = en_core_web_sm.load()

# def get_subject_verb_obj(sentence):
#     print(sentence)
#     tokens = nlp(sentence)
#     svos = findSVOs(tokens)
#     print(svos)
#     print("-------------------------------")

def filter_verbs():
    # with open('data/analyse_verbs/verbs-all.json') as json_file:
    #     verbs = json.load(json_file)
    # list_verbs = []
    # for verb_l in verbs:
    #     list_verbs.append(verb_l[0])
    # print(len(list_verbs))
    #
    # with open('data/all_sentence_transcripts.json') as json_file:
    #     all_sentence_transcripts = json.load(json_file)
    # all_sentences = [l[0] for key in all_sentence_transcripts.keys() for l in all_sentence_transcripts[key]]
    #
    # print(len(all_sentences))
    #
    # filter_out = ["like", "have", "get", "do", "will", "can", "be", "know", "go", "want"]
    # count_verbs = []
    # for sentence in tqdm(list(set(all_sentences))[:10000]):
    #     for verb in list_verbs:
    #         if verb in filter_out:
    #             continue
    #         if verb in sentence.split():
    #             count_verbs.append(verb)
    # most_common = Counter(count_verbs).most_common(200)
    # for l in most_common:
    #     print(l)


    verbs = ["shower", "sleep", "eat", "make", "clean", "dance", "sing", "drink", "buy", "drive",
             "cook", "fall", "read", "write", "study", "play", "paint", "listen", "relax", "wake",
             "walk", "shop", "work", "learn", "cut", "brush", "wash", "run", "throw", "paint", "cool",
             "warm", "give"]

    verbs = list(set(verbs))
    print(len(verbs))

def get_all_verb_dobj(sentence, coref_sentence, list_mentions):
    verbs = ["add", "shower", "sleep", "eat", "make", "clean", "dance", "sing", "drink", "buy", "drive",
             "cook", "fall", "read", "write", "study", "play", "paint", "listen", "relax", "wake",
             "walk", "shop", "work", "learn", "cut", "brush", "wash", "run", "throw", "paint", "cool",
             "warm", "give", "use"]
    tokens = nlp(coref_sentence)
    # list_mentions = ast.literal_eval(list_mentions)
    list_actions = []
    for t in tokens:
        if t.pos_ == "VERB" and t.lemma_ in verbs:
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
            list_actions.append(action)
            # else:
            #     print(action, list_mentions)

    return list_actions

# def get_all_verb_dobj_and_tools(sentence):
#     tokens = nlp(sentence)
#     list_actions = []
#     for t in tokens:
#         if t.pos_ == "VERB":
#             if t.lemma_ not in ["have", "ve", "do", "can", "will", "could", "would", "have", "share", "welcome", "am",
#                                 "be", "was", "were", "let", "love", "like"]:
#                 action = t.lemma_
#                 for tok in t.children:
#                     dep = tok.dep_
#                     if dep == "dobj" or dep == "prt" or dep == "xcomp":
#                         action += (" " + tok.text)
#                 list_actions.append(action)
#
#     return list_actions

def get_all_verbs(sentence):
    tokens = nlp(sentence)
    list_verbs = []
    for t in tokens:
        if t.pos_ == "VERB":
            if t.lemma_ not in ["ve", "do" "can", "will", "could", "would", "have", "share", "welcome", "go", "gonna", "gon", "am", "be", "was", "were", "let"]:
                list_verbs.append(t.lemma_)

    return list_verbs

# def get_context(action, part_sentence):
#     if action in part_sentence:
#

def modify_file():
    with open('data/analyse_verbs/dict_verb_dobj_per_video_specific.json') as json_file:
        dict_verb_dobj_per_video = json.load(json_file)

    video_actions_dict = {}
    video_dict = {"video": []}
    for video in tqdm(list(dict_verb_dobj_per_video.keys())):
        for [verb_dobj, sentence, [time_s, time_e]] in dict_verb_dobj_per_video[video]:
            key = "+".join([video, time_s, time_e]) + ".mp4"
            if key not in video_actions_dict:
                video_actions_dict[key] = []
            if {"action": verb_dobj, "sentence": sentence} not in video_actions_dict[key]:
                video_actions_dict[key].append({"action": verb_dobj, "sentence": sentence})

            if {"time_s": time_s, "time_e": time_e, "video": video} not in video_dict["video"]:
                video_dict["video"].append({"time_s": time_s, "time_e": time_e, "video": video})
        #     if len(video_dict["video"]) == 100:
        #         break
        # break
    with open('data/analyse_verbs/dict_example3.json', 'w+') as fp:
        json.dump(video_dict, fp)
    with open('data/analyse_verbs/dict_example3_actions.json', 'w+') as fp:
        json.dump(video_actions_dict, fp)

def original_utterance_diff(sentence_i, sentence_j, video, i):
    with open('data/all_sentence_transcripts.json') as json_file:
        dict_verb_dobj_per_video_all = json.load(json_file)

    list_data = dict_verb_dobj_per_video_all[video]
    index_i, index_j = 0, 0
    for i in range(len(list_data)):
        if list_data[i][0] == sentence_i:
            index_i = i
        elif list_data[i][0] == sentence_j:
            index_j = i
            break
    return index_j - index_i

def check_if_actions_contained(action1, action2):
    words_1 = set(action1.split())
    words_2 = set(action2.split())
    if words_1.issubset(words_2):
        return action1
    elif words_2.issubset(words_1):
        return action2
    return False

def process_action(action):
    action = action.lower()
    action_words = action.split()
    for word in ["you", "we", "i", "which", "what", "who", "whom", "me", "he", "she", "him", "her", "they", "them",
                 "that", "these", "those", "it"]:
        if word in action_words:
            action_words.remove(word)
    action = " ".join(action_words)
    # if len(action.split()) >= 3:
    #     print(action)
        # action = " ".join(action.split()[0:2])
    return action

def get_all_consecutive_action_pairs():
    # with open('data/analyse_verbs/dict_verb_dobj_per_video_all.json') as json_file:
    # with open('data/analyse_verbs/dict_verb_dobj_per_video_all_coref.json') as json_file:
    with open('data/analyse_verbs/dict_verb_dobj_per_video_all_coref_verbs.json') as json_file:
        dict_verb_dobj_per_video_all = json.load(json_file)

    threshold_utterances = 20 # 3 utterances or time (30 sec)
    remove_verbs = ["give up", "give try", "make time", "use routine", "look", "be", "go", "think", "know", "like", "love", "enjoy", "feel", "do", "have", "last", "show"]

    all_actions = []
    list_all_actions = []
    for video in tqdm(dict_verb_dobj_per_video_all.keys()):
        all_actions_in_video = []
        for [action, sentence, time] in dict_verb_dobj_per_video_all[video]:
            action = process_action(action)
            if action not in remove_verbs and len(action.split()) > 1 and len(action.split()) < 3 and (action.split()[0] != action.split()[1]) and (not action.split()[1].isdigit()) and action.split()[1] not in ["some", "mine", "ooh", "thing", "lot", "account", "whatever", "wake", "much", "many", "top", "sort", "bunch", "all", "avoid", "one", "do", "one", "kind", "it", "a", "thumbs up", "thumbs", "bit", "what", "which", "who", "whom", "difference", "that", "those", "them", "this", "these", "sense", "be", "decision"]:
                video_name = "+".join([video] + time)
                if (action, sentence, video_name) not in all_actions_in_video:
                    all_actions_in_video.append((action, sentence, video_name))
                    list_all_actions.append(action)
        all_actions.append(all_actions_in_video)

    # list_all_actions = list(set(list_all_actions))
    # for i in range(len(list_all_actions)-1):
    #     for j in range(i+1, len(list_all_actions)):
    #         action1, action2 = list_all_actions[i], list_all_actions[j]
    #         check = check_if_actions_contained(action1, action2)
    #         if check:
    #             print(action1 + " ; " + action2 + " ; " + check)



    action_pairs = []
    action_video_pairs_together = {}
    action_video_pairs_not_together = {}
    with open('data/all_sentence_transcripts.json') as json_file:
        dict_verb_dobj_per_video_all = json.load(json_file)

    for list_actions_sentences in tqdm(all_actions):
        for i in range(len(list_actions_sentences) - 1):
            action_i = list_actions_sentences[i][0]
            sentence_i = list_actions_sentences[i][1]
            video_name_i = list_actions_sentences[i][2]
            video = video_name_i.split("+")[0]

            list_data = dict_verb_dobj_per_video_all[video]
            index_x, index_y = 0, 0
            for x in range(len(list_data)):
                if list_data[x][0] == sentence_i:
                    index_x = x
                    break

            for j in range(i + 1, len(list_actions_sentences)):
                # print("here5")
                action_j = list_actions_sentences[j][0]
                sentence_j = list_actions_sentences[j][1]
                video_name_j = list_actions_sentences[j][2]
                for y in range(index_x, len(list_data)):
                    if list_data[y][0] == sentence_j:
                        index_y = y
                        break
                # diff_utterances = original_utterance_diff(sentence_i, sentence_j, video)
                diff_utterances = index_y - index_x
                if diff_utterances <= threshold_utterances:
                    action_pairs.append(str((action_i, action_j)))
                    if (action_i, action_j) not in action_video_pairs_together:
                        action_video_pairs_together[(action_i, action_j)] = []
                    action_video_pairs_together[(action_i, action_j)].append(str([(video_name_i, video_name_j), (sentence_i, sentence_j)]))
                elif diff_utterances >= threshold_utterances + 30 and (action_i, action_j) not in action_video_pairs_together and (action_i != action_j) and (action_i not in action_j) and (action_j not in action_i):
                    if (action_i, action_j) not in action_video_pairs_not_together:
                        action_video_pairs_not_together[(action_i, action_j)] = []
                    action_video_pairs_not_together[(action_i, action_j)].append(str([(video_name_i, video_name_j), (sentence_i, sentence_j)]))

    #filter
    list_all_actions_together = []
    for (a_i, a_j) in action_video_pairs_together.keys():
        list_all_actions_together.append(a_i)
        list_all_actions_together.append(a_j)
    list_all_actions_together = list(set(list_all_actions_together))
    print(len(list_all_actions_together))



    for (a_i, a_j) in tqdm(list(action_video_pairs_not_together.keys())):
        if a_i not in list_all_actions_together or a_j not in list_all_actions_together:
            del action_video_pairs_not_together[(a_i, a_j)]
        if (a_i, a_j) in action_video_pairs_together: #TODO - DONE make sure the actions don't overlap
            del action_video_pairs_not_together[(a_i, a_j)]

    return Counter(action_pairs).most_common(), action_video_pairs_together, action_video_pairs_not_together

def check_if_tuple_inverse(new_action_pairs):
    inverse_tuples = []
    for i in range(len(new_action_pairs)-1):
        action_1, action_2 = new_action_pairs[i][0][0], new_action_pairs[i][0][1]
        for j in range(i+1, len(new_action_pairs)):
            action_3, action_4 = new_action_pairs[j][0][0], new_action_pairs[j][0][1]
            if action_1 == action_4 and action_2 == action_3:
                inverse_tuples.append([action_1, action_2])
                break
    print(inverse_tuples)
    print(len(inverse_tuples))


def filter_not_together_on_distrib(all_pairs_in_video_not_together, all_pairs_in_video_together):
    print("filtering ..")
    print("len(all_pairs_in_video_not_together)", len(all_pairs_in_video_not_together))
    distrib_actions = []
    for data in all_pairs_in_video_together:
        (action_i, action_j) = ast.literal_eval(data)
        distrib_actions.append(action_i)
        distrib_actions.append(action_j)
    distrib_actions1 = Counter(distrib_actions)

    not_together_actions = {}
    distrib_actions2 = []
    #TODO: DONE- Add a threshold to make up for the few examples at action pair level (make sense as those action pairs are not popular)
    threshold = 5
    for action in tqdm(distrib_actions):
        count_action = distrib_actions1[action]
        for (action_i, action_j) in all_pairs_in_video_not_together:
            if action_i == action:
                list_videos_sentences = all_pairs_in_video_not_together[(action_i, action_j)]
                if str((action_j, action_i)) in not_together_actions:
                    not_together_actions[str((action_j, action_i))] += list_videos_sentences
                    not_together_actions[str((action_j, action_i))] = list(set(not_together_actions[str((action_j, action_i))]))
                else:
                    if str((action_i, action_j)) not in not_together_actions:
                        not_together_actions[str((action_i, action_j))] = []
                        distrib_actions2.append(action_i)
                        distrib_actions2.append(action_j)
                    not_together_actions[str((action_i, action_j))] += list_videos_sentences
                    not_together_actions[str((action_i, action_j))] = list(set(not_together_actions[str((action_i, action_j))]))
                    if count_action <= distrib_actions2.count(action_i) - threshold:
                        break
    return not_together_actions

def process_action_pairs():
    action_pairs_together, action_video_pairs_together, action_video_pairs_not_together = get_all_consecutive_action_pairs()
    new_action_pairs = []
    all_actions = []
    my_action_pairs = []
    total_count_pairs = 0
    for data in action_pairs_together:
        actions, count = ast.literal_eval(data[0]), data[1]
        if (actions[0] != actions[1]) and (actions[0] not in actions[1]) and (actions[1] not in actions[0]) and count > 2:
            # new_action_pairs.append((actions, count))
            new_action_pairs.append(actions)
            total_count_pairs += count
            for i in range(count):
                all_actions.append(actions[0])
                all_actions.append(actions[1])
                my_action_pairs.append((actions[0], actions[1]))
                my_action_pairs.append((actions[1], actions[0]))
                # my_verb_pairs.append((actions[0].split()[0], actions[1].split()[0]))
                # my_verb_pairs.append((actions[1].split()[0], actions[0].split()[0]))


    # print(new_action_pairs)
    print(len(new_action_pairs))
    print(total_count_pairs)

    total_count_videos_pairs = 0
    together_actions = {}
    for together_action_pairs in new_action_pairs:
        (action_i, action_j) = together_action_pairs
        list_videos_sentences = action_video_pairs_together[together_action_pairs]
        if str((action_j, action_i)) in together_actions:  #TODO - DONE: MAKE SURE THE ORDER DOESN'T MATTER
            together_actions[str((action_j, action_i))] += list_videos_sentences
            together_actions[str((action_j, action_i))] = list(set(together_actions[str((action_j, action_i))]))
        else:
            together_actions[str(together_action_pairs)] = list(set(list_videos_sentences))
        # together_actions[str(together_action_pairs)] = list(set(list_videos_sentences))
        total_count_videos_pairs += len(set(list_videos_sentences))

    assert(total_count_videos_pairs == total_count_pairs)
    with open('data/final_data/dict_together_actions.json', 'w+') as fp:
        json.dump(together_actions, fp)

    # check_if_tuple_inverse(new_action_pairs)

    # verbs = []
    # for action in all_actions:
    #     verbs.append(action.split()[0])
    # all_verbs = list(set(verbs))
    # print(str(len(all_verbs)) + " verbs")
    # # print(Counter(verbs).most_common())  #TODO? - REMOVE fall, walk, sleep, warm

    ###filter diff based on distribution from my_action_pairs
    not_together_actions = filter_not_together_on_distrib(action_video_pairs_not_together, list(together_actions.keys()))
    with open('data/final_data/dict_not_together_actions.json', 'w+') as fp:
        json.dump(not_together_actions, fp)

def get_all_together_actions():
    with open('data/final_data/dict_together_actions.json') as json_file:
        dict_together_actions = json.load(json_file)

    print("-------Together actions: -------")
    list_together_tuples = []
    count_total_pairs = 0
    list_actions = []
    list_tuple_actions = []
    for key in dict_together_actions:
        count_total_pairs += len(dict_together_actions[key])
        list_together_tuples.append(key)
        list_actions.append(ast.literal_eval(key)[0])
        list_actions.append(ast.literal_eval(key)[1])
        list_tuple_actions.append(ast.literal_eval(key))
    list_actions = list(set(list_actions))
    print(list_actions)
    print("count_pairs", len(dict_together_actions.keys()))
    print("count_total_data", count_total_pairs)
    print("count_actions", len(list_actions))
    # print(list_tuple_actions)
    return list_actions

def check_final_data():
    with open('data/final_data/dict_together_actions.json') as json_file:
        dict_together_actions = json.load(json_file)
    with open('data/final_data/dict_not_together_actions.json') as json_file:
        dict_not_together_actions = json.load(json_file)

    print("-------Together actions: -------")
    list_together_tuples = []
    count_total_pairs = 0
    for key in dict_together_actions:
        count_total_pairs += len(dict_together_actions[key])
        list_together_tuples.append(key)
    print("count_pairs", len(dict_together_actions.keys()))
    print("count_total_data", count_total_pairs)

    print("--------Not Together actions: -----------")
    list_not_together_tuples = []
    count_total_pairs = 0
    for key in dict_not_together_actions:
        count_total_pairs += len(dict_not_together_actions[key])
        list_not_together_tuples.append(key)
    print("count_pairs", len(dict_not_together_actions.keys()))
    print("count_total_data", count_total_pairs)

    assert(list(set(list_together_tuples) & set(list_not_together_tuples)) == []) #no actions overlap
    list_actions = []
    for data in list_together_tuples:
        (a_i, a_j) = ast.literal_eval(data)
        list_actions.append(a_i)
        list_actions.append(a_j)
        if str((a_j, a_i)) in list_together_tuples:
            raise ValueError("error")
    for data in list_not_together_tuples:
        (a_i, a_j) = ast.literal_eval(data)
        list_actions.append(a_i)
        list_actions.append(a_j)
        if str((a_j, a_i)) in list_not_together_tuples:
            raise ValueError("error")

    # print(Counter(list_actions).most_common())



def get_verbs_all_videos():
    # with open('data/all_sentence_transcripts.json') as json_file:
    with open('data/coref_all_sentence_transcripts.json') as json_file:
    # with open('data/UCM3P_G21gOSVdepXrEFojIg.json') as json_file:
        all_sentence_transcripts_rachel = json.load(json_file)

    dict_verb_dobj_per_video = {}
    dict_verb_dobj = {}
    print(len(all_sentence_transcripts_rachel.keys()))
    for video in tqdm(list(all_sentence_transcripts_rachel.keys())):
        actions_per_video = []
        for dict in list(all_sentence_transcripts_rachel[video]):
            part_sentence, coref_sentence, time_s, time_e, list_mentions = dict["sentence"], dict["coref_sentence"], \
                                                                           dict["time_s"], dict["time_e"], dict["list_mentions"]
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
    with open('data/analyse_verbs/dict_verb_dobj_per_video_all_coref_verbs.json', 'w+') as fp:
        json.dump(dict_verb_dobj_per_video, fp)

def analyse_verbs():
    with open('analyse_verbs/dict_verbs_per_video_rachel.json') as json_file:
        dict_verbs_per_video = json.load(json_file)

    # verb = "clean"
    for verb in ["clean", "read", "write", "play"]:
        before_verb = []
        after_verb = []
        for video in dict_verbs_per_video.keys():
            list_verbs = dict_verbs_per_video[video]
            indices_verb = [i for i, x in enumerate(list_verbs) if x == verb]
            if indices_verb:
                for ind in indices_verb:
                    for v in list_verbs[ind-3:ind]:
                        before_verb.append(v)
                    for v in list_verbs[ind+1:ind+4]:
                        after_verb.append(v)
                    # print("before " + list_verbs[ind] + ": " + str(list_verbs[ind-3:ind]))
                    # print("after " + list_verbs[ind] + ": " + str(list_verbs[ind+1:ind+4]))

        print(verb)
        # print(Counter(before_verb).most_common()[-10:])
        print(Counter(before_verb).most_common()[:15])
        print(Counter(after_verb).most_common()[:15])

def analyse_actions():
    with open('data/analyse_verbs/dict_verb_dobj_per_video_specific.json') as json_file:
        dict_verb_dobj_per_video = json.load(json_file)

    # verb = "clean"
    # for verb in ["clean", "read", "write", "play"]:
    for verb in ["clean"]:
        before_verb = []
        after_verb = []
        for video in dict_verb_dobj_per_video.keys():
            list_verbs = dict_verb_dobj_per_video[video]
            indices_verb = [i for i, x in enumerate(list_verbs) if x == verb]
            if indices_verb:
                for ind in indices_verb:
                    ind_index_before = 1
                    ind_index_after = 1
                    nb_before = 0
                    nb_after = 0
                    while nb_before < 3:
                        action_before = list_verbs[ind - ind_index_before]
                        if len(action_before.split()) >= 2:
                            before_verb.append(action_before)
                        ind_index_before += 1
                        nb_before += 1
                    while nb_after < 3:
                        action_after = list_verbs[ind + ind_index_after]
                        if len(action_after.split()) >= 2:
                            after_verb.append(action_after)
                        ind_index_after += 1
                        nb_after += 1
                    # for v in list_verbs[ind-3:ind]:
                    #     before_verb.append(v)
                    # for v in list_verbs[ind+1:ind+4]:
                    #     after_verb.append(v)
                    # print("before " + list_verbs[ind] + ": " + str(list_verbs[ind-3:ind]))
                    # print("after " + list_verbs[ind] + ": " + str(list_verbs[ind+1:ind+4]))

        print(verb)
        # print(Counter(before_verb).most_common()[-10:])
        print(Counter(before_verb).most_common())
        print(Counter(after_verb).most_common())

def analyse_actions2():
    with open('data/analyse_verbs/dict_verb_dobj_per_video_specific.json') as json_file:
        dict_verb_dobj_per_video = json.load(json_file)


        for video in list(dict_verb_dobj_per_video.keys())[:10]:
            list_verbs = dict_verb_dobj_per_video[video]
            list_verb_obj = []
            for verb in list_verbs:
                if len(verb.split()) >= 2:
                    list_verb_obj.append(verb)
            print(list_verb_obj)
            print("-------------")


def analyse_file():
    with open('data/analyse_verbs/dict_example3_actions.json') as json_file:
        dict_example3 = json.load(json_file)

    video_dict = {}
    for video in dict_example3.keys():
        only_video = video.split("+")[0]
        if only_video not in video_dict:
            video_dict[only_video] = []
        for l in dict_example3[video]:
            video_dict[only_video].append(l["action"])

    with open('data/analyse_verbs/dict_example3_analyse.json', 'w+') as fp:
        json.dump(video_dict, fp)

def cluster_actions():
    list_actions = get_all_together_actions()
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



def main():
    # get_verbs_all_videos()

    # modify_file()
    # analyse_file()
    # filter_verbs()

    # get_all_consecutive_action_pairs()
    # process_action_pairs()
    # check_final_data()
    # get_all_together_actions()
    cluster_actions()

    # sentence = "this stew is very like flexitarian you could leave it vegetarian not add the sausage"
    # list_actions = get_all_verb_dobj(sentence)
    # print(list_actions)
    # analyse_verbs()
    # analyse_actions()
    # analyse_actions2()

if __name__ == '__main__':
    main()
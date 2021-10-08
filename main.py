import json
import en_core_web_sm
from tqdm import tqdm
from collections import Counter

nlp = en_core_web_sm.load()

def read_sentence_transcripts():
    with open('data/UCM3P_G21gOSVdepXrEFojIg.json') as json_file:
        data = json.load(json_file)

    print(len(data.keys()))
    dict_sentences = {}
    for key in data.keys():
        list_sentences = []
        for [sentence, start, end] in data[key]:
            list_sentences.append(sentence)
        dict_sentences[key] = list_sentences

    with open('data/sentences_UCM3P_G21gOSVdepXrEFojIg.json', 'w+') as fp:
        json.dump(dict_sentences, fp)

def get_verb_pairs():

    with open('data/analyse_verbs/dict_verb_dobj_per_video.json') as json_file:
        data = json.load(json_file)

    list_pairs = []

    verbs_remove = ['like', 'want', 'know', 'feel', 'try', 'think', 'try', 'may', 'get']
    for video in data.keys():
        list_actions = data[video]
        list_actions_filtered = [action for action in list_actions if action not in verbs_remove and len(action.split()) >= 2]
        for i in range(len(list_actions_filtered)-1):
            action1 = list_actions_filtered[i]
            action2 = list_actions_filtered[i+1]

            list_pairs.append(str([action1, action2]))

    for i in Counter(list_pairs).most_common(100):
        print(i)
    # print(Counter(list_pairs).most_common(20))
if __name__ == '__main__':
    # read_sentence_transcripts()
    get_verb_pairs()


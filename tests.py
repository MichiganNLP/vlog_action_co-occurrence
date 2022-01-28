import json
import ast
from collections import Counter
import spacy
import pandas as pd
from tqdm.auto import tqdm
from main import get_VP

nlp = spacy.load("en_core_web_trf")


def try_recursion():
    from spacy import displacy
    coref_sentence = "so i used like two tablespoons of dry oregano oregano"
    print(coref_sentence)
    tokens = nlp(coref_sentence)
    displacy.serve(tokens, style="dep")

    list_actions = []
    for t in tokens:
        if t.pos_ == "VERB":
            action = get_VP(t, [t])
            list_actions.append(" ".join([tok.lemma_ for tok in sorted(action, key=lambda tok: tok.i)]))
    print(list_actions)


def compute_PMI():
    with open('data/dict_verb_dobj_per_video_all_coref_verbs.json') as json_file:
        dict_verb_dobj_per_video = json.load(json_file)
    all_actions = []
    for video in tqdm(dict_verb_dobj_per_video):
        for action_data in dict_verb_dobj_per_video[video]:
            all_actions.append(ast.literal_eval(action_data)[0])


def compute_concretness():
    with open('data/dict_verb_dobj_per_video_all_coref_verbs.json') as json_file:
        dict_verb_dobj_per_video = json.load(json_file)
    all_actions = []
    for video in tqdm(dict_verb_dobj_per_video):
        for action_data in dict_verb_dobj_per_video[video]:
            all_actions.append(ast.literal_eval(action_data)[0])

    df = pd.read_csv('data/utils/concretness.txt', sep="	",
                     names=['Word', 'Bigram', 'Conc.M', 'Conc.SD', 'Unknown', 'Total', 'Percent_known', 'SUBTLEX',
                            'Dom_Pos'])
    df.dropna(inplace=True)

    list_words = list(df["Word"][1:].array)
    list_scores = list(df["Conc.M"][1:].array)
    count_found = 0
    list_low_scores = []
    list_high_scores = []
    for a_c in tqdm(Counter(all_actions).most_common()):
        action = a_c[0]
        found = False
        for (word, score) in zip(list_words, list_scores):
            score = float(score)
            if action == word:
                # print(action, score)
                count_found += 1
                found = True
                if score >= 3:
                    list_high_scores.append((action, score))
                else:
                    list_low_scores.append((action, score))
                break
        if not found:
            for (word, score) in zip(list_words, list_scores):
                score = float(score)
                if action.split()[0] == word or action.split()[1] == word or action.split()[1] in word.split():
                    if score >= 2:
                        list_high_scores.append((action, score))
                    else:
                        list_low_scores.append((action, score))
                    count_found += 1
                    break
    with open('data/utils/results_concretness.txt', 'w') as f:
        f.write("%s\n" % "------------------------ High scores")
        for item in list_high_scores:
            f.write("%s\n" % str(item))
        f.write("%s\n" % "------------------------ Low scores")
        for item in list_low_scores:
            f.write("%s\n" % str(item))

    print(count_found, len(list(set(all_actions))))
    print(len(list_high_scores), len(list(list_low_scores)))


if __name__ == '__main__':
    try_recursion()

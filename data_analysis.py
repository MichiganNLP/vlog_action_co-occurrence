import json
from collections import defaultdict
import ast

import numpy as np
from rich.progress import track
import math


def get_all_action_pairs(dict_video_action_pairs_filtered):
    all_action_pairs = [str(sorted((action_1, action_2)))
                        for video in dict_video_action_pairs_filtered
                        for (action_1, transcript_a1, clip_a1), (action_2, transcript_a2, clip_a2)
                        in dict_video_action_pairs_filtered[video]]

    all_action_pairs = [tuple(ast.literal_eval(action_pair)) for action_pair in all_action_pairs]
    return all_action_pairs


def get_all_actions(all_action_pairs):
    all_actions = []
    for [a1, a2] in all_action_pairs:
        all_actions.append(a1)
        all_actions.append(a2)
    return all_actions


def calculate_PMI(all_actions, all_action_pairs, file_out):
    N = len(all_actions)
    M = len(all_action_pairs)
    threshold = 10
    pmi_action_pairs = defaultdict()
    for action_pair in track(set(all_action_pairs), description="Computing PMI..."):
        (a1, a2) = action_pair
        if all_actions.count(a1) < threshold or all_actions.count(a2) < threshold:
            pmi_action_pairs[str((a1, a2))] = 0
        else:
            # weighting PMI: • Raise the context probabilities to alpha = 0.75
            # http://www.cs.umd.edu/class/fall2018/cmsc470/slides/slides_02.pdf
            # alpha = 0.75
            norm_count_a1_a2 = all_action_pairs.count((a1, a2)) / M
            norm_count_a1 = all_actions.count(a1) / N
            norm_count_a2 = all_actions.count(a2) / N
            # norm_count_a1 = all_actions.count(a1) ** alpha / N
            # norm_count_a2 = all_actions.count(a2) ** alpha / N
            PMI_a1_a2 = max(0, int(math.log2(norm_count_a1_a2 / (norm_count_a1 * norm_count_a2))))
            pmi_action_pairs[str((a1, a2))] = PMI_a1_a2

    with open(file_out, 'w+') as fp:
        json.dump(pmi_action_pairs, fp)


def analyse_results():
    with open('data/results/labels_test.npy', 'rb') as f:
        labels_test = np.load(f, allow_pickle=True)
    with open('data/results/nodes_test.npy', 'rb') as f:
        nodes_test = np.load(f, allow_pickle=True)
    with open('data/results/predicted_SVM_all_txt_vis_embeddings_all_heuristics.npy', 'rb') as f:
        predicted_labels = np.load(f)
    with open('data/dict_video_action_pairs_filtered_by_link.json') as json_file:
        dict_video_action_pairs_filtered = json.load(json_file)

    all_action_pairs = get_all_action_pairs(dict_video_action_pairs_filtered)
    all_actions = get_all_actions(all_action_pairs)

    nodes_name_pairs = [(nodes_test[i][0], nodes_test[i][1]) for i in range(len(nodes_test))]

    dict_results_correct = defaultdict()
    dict_results_errors = {"FP": {}, "FN": {}}
    for i, (predicted, gt) in enumerate(zip(predicted_labels, labels_test)):
        if predicted == gt:
            dict_results_correct[nodes_name_pairs[i]] = all_actions.count(nodes_name_pairs[i][0]) + all_actions.count(
                nodes_name_pairs[i][1])
        elif predicted == 0 and gt == 1:
            dict_results_errors["FN"][nodes_name_pairs[i]] = all_actions.count(
                nodes_name_pairs[i][0]) + all_actions.count(nodes_name_pairs[i][1])
        else:
            dict_results_errors["FP"][nodes_name_pairs[i]] = all_actions.count(
                nodes_name_pairs[i][0]) + all_actions.count(nodes_name_pairs[i][1])

    print("#FN", str(len(dict_results_errors["FN"])))
    print(dict(sorted(dict_results_errors["FN"].items()), key=lambda item: item[0]))
    print("#FP", str(len(dict_results_errors["FP"])))
    print(dict(sorted(dict_results_errors["FP"].items()), key=lambda item: item[0]))
    print("#TP + TN", len(dict_results_correct))


def main():
    with open('data/dict_video_action_pairs_filtered_by_link.json') as json_file:
        dict_video_action_pairs_filtered = json.load(json_file)

    all_action_pairs = get_all_action_pairs(dict_video_action_pairs_filtered)
    all_actions = get_all_actions(all_action_pairs)
    all_verbs = [action.split()[0] for action in all_actions]
    all_verb_pairs = [(action_pair[0].split()[0], action_pair[1].split()[0]) for action_pair in all_action_pairs]
    calculate_PMI(all_actions, all_action_pairs, file_out='data/dict_pmi_action_pairs.json')
    calculate_PMI(all_verbs, all_verb_pairs, file_out='data/dict_pmi_verb_pairs.json')

    analyse_results()


if __name__ == '__main__':
    main()

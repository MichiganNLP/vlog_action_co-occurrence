from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import json
from collections import Counter, defaultdict
import ast
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from rich.progress import track
from tqdm import tqdm
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
    pmi_action_pairs = defaultdict()
    for action_pair in track(set(all_action_pairs), description="Computing PMI..."):
        (a1, a2) = action_pair
        # if all_actions.count(a1) < 50 or all_actions.count(a2) < 50:
        #     pmi_action_pairs[str((a1, a2))] = 'None'
        # else:
        # weighting PMI: • Raise the context probabilities to alpha = 0.75
        # http://www.cs.umd.edu/class/fall2018/cmsc470/slides/slides_02.pdf
        alpha = 0.75
        norm_count_a1_a2 = all_action_pairs.count((a1, a2)) / M
        # norm_count_a1 = all_actions.count(a1) / N
        # norm_count_a2 = all_actions.count(a2) / N
        norm_count_a1 = all_actions.count(a1) ** alpha / N
        norm_count_a2 = all_actions.count(a2) ** alpha / N
        PMI_a1_a2 = int(math.log2(norm_count_a1_a2 / (norm_count_a1 * norm_count_a2)))
        pmi_action_pairs[str((a1, a2))] = PMI_a1_a2

    with open(file_out, 'w+') as fp:
        json.dump(pmi_action_pairs, fp)

def main():
    with open('data/dict_video_action_pairs_filtered_by_link.json') as json_file:
        dict_video_action_pairs_filtered = json.load(json_file)

    all_action_pairs = get_all_action_pairs(dict_video_action_pairs_filtered)
    all_actions = get_all_actions(all_action_pairs)
    all_verbs = [action.split()[0] for action in all_actions]
    # all_obj = [action.split()[1] for action in all_actions]
    all_verb_pairs = [(action_pair[0].split()[0], action_pair[1].split()[0]) for action_pair in all_action_pairs]
    calculate_PMI(all_actions, all_action_pairs, file_out='data/dict_pmi_action_pairs.json')
    calculate_PMI(all_verbs, all_verb_pairs, file_out='data/dict_pmi_verb_pairs.json')
    # calculate_PMI(all_obj, all_obj_pairs, file_out='data/dict_pmi_obj_pairs.json')

if __name__ == '__main__':
    main()
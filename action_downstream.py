import json
from collections import Counter
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import numpy as np
from rich.progress import track
import pandas as pd

from data_processing import get_sentence_embedding_features
from link_prediction import test_my_data


def read_dataset_actions(dataset):
    set_action_train = set()
    print(f"Reading {dataset} ...")
    with open(f"data/action_localization/{dataset}.json") as f:
        dict_coin = json.loads(f.read())
    time_difference = 10 #COIN, TODO: other datasets need different?
    dict_video_action_pairs = {}
    dict_action_location_nn = {}
    for video in tqdm(list(dict_coin['database'].keys())):
        dict_video_action_pairs[video] = []
        annotations_video = dict_coin['database'][video]["annotation"]
        data_type = dict_coin['database'][video]["subset"]
        if data_type == "testing":#validation - youcook2
            actions_in_time_order = [annotation['label'] for annotation in annotations_video] #sentence
            for i, action in enumerate(actions_in_time_order[1:-1]):
                if action not in dict_action_location_nn:
                    dict_action_location_nn[action] = []
                dict_action_location_nn[action].append(actions_in_time_order[i])
                dict_action_location_nn[action].append(actions_in_time_order[i+2])
            # first and last actions have only 1 neighbouring action in the video
            first_action, last_action = actions_in_time_order[0], actions_in_time_order[-1]
            if first_action not in dict_action_location_nn:
                dict_action_location_nn[first_action] = []
            if last_action not in dict_action_location_nn:
                dict_action_location_nn[last_action] = []
            dict_action_location_nn[first_action].append(actions_in_time_order[1])
            dict_action_location_nn[last_action].append(actions_in_time_order[-2])

        if data_type != "training":
            continue
        actions_time = []
        for annotation in annotations_video:
            time = annotation['segment']
            action = annotation['label'] #sentence
            actions_time.append([action, time])
            set_action_train.add(action)

        for i in range(0, len(actions_time) - 1):
            for j in range(i + 1, len(actions_time)):
                difference = actions_time[j][1][0] - actions_time[i][1][1]
                if difference > time_difference:
                    break
                action_1, clip_a1 = actions_time[i]
                action_2, clip_a2 = actions_time[j]
                if [(action_1, clip_a1), (action_2, clip_a2)] not in dict_video_action_pairs[video] \
                        and [(action_2, clip_a2), (action_1, clip_a1)] not in dict_video_action_pairs[video]:
                    dict_video_action_pairs[video].append([(action_1, clip_a1), (action_2, clip_a2)])

    with open(f'data/dict_video_action_pairs_{dataset}.json', 'w+') as fp:
        json.dump(dict_video_action_pairs, fp)
    with open(f'data/dict_action_location_nn_{dataset}.json', 'w+') as fp:
        json.dump(dict_action_location_nn, fp)

def get_graph(dataset):
    print(f"Creating graph for {dataset} ...")
    with open(f'data/dict_video_action_pairs_{dataset}.json') as json_file:
        dict_video_action_pairs= json.load(json_file)
    action_pairs = [sorted((action_1, action_2))
                    for video in dict_video_action_pairs
                    for (action_1, clip_a1), (action_2, clip_a2)
                    in dict_video_action_pairs[video]]

    actions = set()
    for action_pair in action_pairs:
        actions.add(action_pair[0])
        actions.add(action_pair[1])
    actions = sorted(actions)
    list_stsbrt_embeddings = get_sentence_embedding_features(actions)
    '''
        Save graph nodes
    '''
    df = pd.DataFrame([tensor.cpu().numpy() for tensor in list_stsbrt_embeddings], index=actions)
    df.to_csv(f'data/graph/txt_action_nodes_{dataset}.csv')
    '''
           Save graph edges
    '''
    counter = Counter(tuple(x) for x in action_pairs)
    list_tuples_actions = [(action_pair[0], action_pair[1], counter[action_pair]) for action_pair in counter]
    df = pd.DataFrame(list_tuples_actions, columns=['source', 'target', 'weight'])
    df.to_csv(f'data/graph/edges_{dataset}.csv', index=False)

def get_graph_embeddings(dataset):
    print(f"Creating graph embeddings for {dataset} ...")
    g_train = test_my_data(input_nodes=f'data/graph/txt_action_nodes_{dataset}.csv', input_edges=f'data/graph/edges_{dataset}.csv')
    list_weighted_avg_embeddings = []
    self_weight = 1
    for node in track(g_train.nodes(), description=f"Computing {dataset} graph embeddings from weighted avg of neighbours..."):
        node_emb_weighted = g_train.node_features(nodes=[node]) * self_weight
        sum_weights = self_weight
        for node_neighbour, edge_weight in g_train.in_nodes(node, include_edge_weight=True):
            # edge_weight = 1 #TODO: replace with non-weighted?
            node_emb_weighted += g_train.node_features(nodes=[node_neighbour]) * edge_weight
            sum_weights += edge_weight
        list_weighted_avg_embeddings.append(node_emb_weighted / sum_weights)  # weighted edge mean of neighbour nodes

    df = pd.DataFrame([np.squeeze(tensor) for tensor in list_weighted_avg_embeddings], index=list(g_train.nodes()))
    df.to_csv(f'data/graph/graph_txt_action_nodes_{dataset}.csv')

def compare_graphs(dataset):
    vlog_node_data = pd.read_csv('data/graph/txt_action_nodes.csv', index_col=0)
    coin_node_data = pd.read_csv(f'data/graph/txt_action_nodes_{dataset}.csv', index_col=0)

    txt_action_node_data_values_coin = coin_node_data.values
    txt_action_node_data_values_vlog = vlog_node_data.values
    action_coin = coin_node_data.index.values.tolist()
    action_vlog = vlog_node_data.index.values.tolist()

    nb_same = 0
    for check_action, action_emb in zip(action_coin, txt_action_node_data_values_coin):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(txt_action_node_data_values_vlog)
        index_action = action_coin.index(check_action)
        check_action_features = txt_action_node_data_values_coin[index_action].reshape(1, -1)
        distance, list_indexes = knn.kneighbors(check_action_features, return_distance=True)
        neighbours = [action_vlog[index] for index in list_indexes[0]]
        if distance[0][0] <= 16:
            nb_same += 1
        # print(f"Graph Neighbours for: {check_action}: {neighbours}, dist:{distance[0][0]}")
    print(nb_same, len(action_coin))

def main():
    for dataset in ["EpicKitchens", "Breakfast", "COIN"]:
        read_dataset_actions(dataset)
        get_graph(dataset)
        get_graph_embeddings(dataset)
        # compare_graphs(dataset)

if __name__ == '__main__':
    main()
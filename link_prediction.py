import argparse
import json
import math
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sknetwork.linkpred import CommonNeighbors, JaccardIndex, SaltonIndex, HubPromotedIndex, AdamicAdar, \
    ResourceAllocation, PreferentialAttachment, HubDepressedIndex, whitened_sigmoid
from sknetwork.path import shortest_path
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter, BiasedRandomWalk, UnsupervisedSampler
from stellargraph.layer import GCN, LinkEmbedding, Node2Vec, Attri2Vec, GraphSAGE
from stellargraph.mapper import FullBatchLinkGenerator, Node2VecLinkGenerator, Node2VecNodeGenerator, \
    Attri2VecLinkGenerator, Attri2VecNodeGenerator, GraphSAGELinkGenerator, GraphSAGENodeGenerator
from tensorflow import keras
from rich.progress import track
from rich.console import Console

random_seed = 10
tf.keras.utils.set_random_seed(
    random_seed
)

console = Console()

''' Hyperparameters '''
# walk_length, epochs, batch_size = 5, 6, 50
walk_length, epochs, batch_size = 10, 6, 50


def create_biased_random_walker(graph, walk_num, walk_length):
    p = 0.5  # 1 defines probability, 1/p, of returning to source node
    q = 1.0  # 1 defines probability, 1/q, for moving to a node away from the source node
    return BiasedRandomWalk(graph, n=walk_num, length=walk_length, p=p, q=q)


def test_my_data(input_nodes, input_edges):
    square_weight_edges = pd.read_csv(input_edges)
    square_node_data = pd.read_csv(input_nodes, index_col=0)
    g = StellarGraph(
        {"action": square_node_data}, {"co-occurs": square_weight_edges}
    )
    print(g.info())
    return g


def test_val_train_split(g):
    # Define an edge splitter on the original graph g:
    edge_splitter_test = EdgeSplitter(g)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from g, and obtain the
    # reduced graph g_test with the sampled links removed:
    g_test, nodes_test, labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global", keep_connected=True, seed=random_seed
    )

    # Define an edge splitter on the reduced graph g_test:
    edge_splitter_val = EdgeSplitter(g_test)

    g_val, nodes_val, labels_val = edge_splitter_val.train_test_split(
        p=0.1, method="global", seed=random_seed
    )

    # Define an edge splitter on the reduced graph g_test:
    edge_splitter_train = EdgeSplitter(g_val)

    g_train, nodes_train, labels_train = edge_splitter_train.train_test_split(
        p=0.1, method="global", seed=random_seed
    )

    # print(g_train.info())
    # print(g_val.info())
    # print(g_test.info())
    # print(len(nodes_test), len(nodes_train), len(nodes_val))
    return g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test


def node_strength(all_edge_info, node_z):
    strength = 0
    source, target, weights = all_edge_info[0].tolist(), all_edge_info[1].tolist(), all_edge_info[3].tolist()
    for (s, t, w) in zip(source, target, weights):
        if s == node_z or t == node_z:
            strength += w
    return strength


# using: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4744029/
def compute_weighted_heuristic_methods(g_val, nodes_val, nodes_test):
    nodes_val = [list([nodes_val[i][0], nodes_val[i][1]]) for i in range(len(nodes_val))]
    nodes_test = [list([nodes_test[i][0], nodes_test[i][1]]) for i in range(len(nodes_test))]
    all_nodes = nodes_val + nodes_test
    all_edge_info = g_val.edge_arrays(include_edge_weight=True)
    source, target, weights = all_edge_info[0].tolist(), all_edge_info[1].tolist(), all_edge_info[3].tolist()

    dict_common_neighbours = defaultdict()
    dict_common_neighbours_w = defaultdict()
    for (node1, node2) in track(all_nodes, description="Computing WCN, WAA, WRA.."):
        common_neighbours = set(g_val.in_nodes(node1)).intersection(set(g_val.in_nodes(node2)))
        dict_common_neighbours[(node1, node2)] = common_neighbours
        WCN, WAA, WRA = 0, 0, 0
        for node_z in common_neighbours:
            weight_node_z = 0
            strength_node_z = node_strength(all_edge_info, node_z)
            for (s, t, w) in zip(source, target, weights):
                if (node1 == s and node_z == t) or (node2 == s and node_z == t) or (node1 == t and node_z == s) or (
                        node2 == t and node_z == s):
                    weight_node_z += w
            WCN += weight_node_z
            WAA += weight_node_z / math.log(1 + strength_node_z)
            WRA += weight_node_z / strength_node_z

        dict_common_neighbours_w[(node1, node2)] = {"WCN": WCN, "WAA": WAA, "WRA": WRA}

    return dict_common_neighbours_w


def weighted_heuristic_methods(nodes_test, labels_test, g_val, nodes_val, labels_val):
    dict_common_neighbours_w = compute_weighted_heuristic_methods(g_val, nodes_val, nodes_test)
    nodes_val = [list([nodes_val[i][0], nodes_val[i][1]]) for i in range(len(nodes_val))]
    nodes_test = [list([nodes_test[i][0], nodes_test[i][1]]) for i in range(len(nodes_test))]

    list_sim_predicted_WCN_val, list_sim_predicted_WCN_test = [], []
    list_sim_predicted_WAA_val, list_sim_predicted_WAA_test = [], []
    list_sim_predicted_WRA_val, list_sim_predicted_WRA_test = [], []
    for (node1, node2) in nodes_val:
        list_sim_predicted_WCN_val.append(dict_common_neighbours_w[(node1, node2)]["WCN"])
        list_sim_predicted_WAA_val.append(dict_common_neighbours_w[(node1, node2)]["WAA"])
        list_sim_predicted_WRA_val.append(dict_common_neighbours_w[(node1, node2)]["WRA"])

    for (node1, node2) in nodes_test:
        list_sim_predicted_WCN_test.append(dict_common_neighbours_w[(node1, node2)]["WCN"])
        list_sim_predicted_WAA_test.append(dict_common_neighbours_w[(node1, node2)]["WAA"])
        list_sim_predicted_WRA_test.append(dict_common_neighbours_w[(node1, node2)]["WRA"])

    for method_name in ["WCN", "WAA", "WRA"]:
        if method_name == "WCN":
            list_sim_predicted_val = list_sim_predicted_WCN_val
            list_sim_predicted_test = list_sim_predicted_WCN_test
        elif method_name == "WAA":
            list_sim_predicted_val = list_sim_predicted_WAA_val
            list_sim_predicted_test = list_sim_predicted_WAA_test
        elif method_name == "WRA":
            list_sim_predicted_val = list_sim_predicted_WRA_val
            list_sim_predicted_test = list_sim_predicted_WRA_test
        else:
            raise ValueError(f"Error with method name, not one of WCN, WAA or WRA")

        max_accuracy, max_threshold = 0, 0
        for threshold in np.linspace(0, 1, 10).tolist():
            # predicted = whitened_sigmoid(np.asarray(list_sim_predicted_val)) > threshold  # keep whitened sigmoid?
            predicted = np.asarray(list_sim_predicted_val) > threshold
            accuracy = accuracy_score(labels_val, predicted)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_threshold = threshold
        max_accuracy = max_accuracy * 100
        console.print(
            f"Method {method_name}, on validation max accuracy: {max_accuracy:.1f} with threshold: {max_threshold:.2f}",
            style="magenta")

        predicted = whitened_sigmoid(np.asarray(list_sim_predicted_test)) > max_threshold
        accuracy = accuracy_score(labels_test, predicted) * 100
        console.print(
            f"Method {method_name}, on test accuracy: {accuracy:.1f} with threshold: {max_threshold:.2f}",
            style="magenta")
        print(Counter(predicted))


def finetune_threshold_on_validation(g_val, nodes_val, labels_val):
    dict_method_threshold = {}
    for method_name in ["CommonNeighbors", "JaccardIndex", "SaltonIndex", "PreferentialAttachment", "AdamicAdar",
                        "HubPromotedIndex", "HubDepressedIndex", "ResourceAllocation", "ShortestPath"]:
        if method_name == "CommonNeighbors":
            method = CommonNeighbors()
        elif method_name == "JaccardIndex":
            method = JaccardIndex()
        elif method_name == "SaltonIndex":
            method = SaltonIndex()
        elif method_name == "PreferentialAttachment":
            method = PreferentialAttachment()
        elif method_name == "HubPromotedIndex":
            method = HubPromotedIndex()
        elif method_name == "HubDepressedIndex":
            method = HubDepressedIndex()
        elif method_name == "AdamicAdar":
            method = AdamicAdar()
        elif method_name == "ResourceAllocation":
            method = ResourceAllocation()
        elif method_name == "ShortestPath":
            pass
        else:
            raise ValueError(f"method {method_name} nam not correct")

        adjacency = g_val.to_adjacency_matrix()
        if method_name != "ShortestPath":
            method.fit_predict(adjacency, 0)  # assigns a scores to edges

        nodes_pairs_val = [list(g_val.node_ids_to_ilocs([nodes_val[i][0], nodes_val[i][1]])) for i in
                           range(len(nodes_val))]

        list_sim_predicted = []
        for (node1, node2) in nodes_pairs_val:
            if method_name != "ShortestPath":
                common_neighbour_similarity = method.predict((node1, node2))
            else:
                list_shortest_path = shortest_path(adjacency, node1, node2)
                if list_shortest_path:
                    common_neighbour_similarity = 1 / len(list_shortest_path)
                else:
                    common_neighbour_similarity = 0
            list_sim_predicted.append(common_neighbour_similarity)

        max_accuracy, max_threshold = 0, 0
        for threshold in np.linspace(0, 1, 10).tolist():
            # predicted = whitened_sigmoid(np.asarray(list_sim_predicted)) > threshold  # keep whitened sigmoid?
            predicted = np.asarray(list_sim_predicted) > threshold
            accuracy = accuracy_score(labels_val, predicted)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_threshold = threshold
        dict_method_threshold[method_name] = max_threshold
        max_accuracy = max_accuracy * 100
        console.print(
            f"Method {method_name}, on validation max accuracy: {max_accuracy:.1f} with threshold: {max_threshold:.2f}",
            style="magenta")
    return dict_method_threshold


def plot_features(method, top_features, feature_names):  # TODO
    import matplotlib.pyplot as plt
    coef = method.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    plt.figure(figsize=(18, 7))
    colors = ['green' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1 + 2 * top_features), feature_names[top_coefficients], rotation=45, ha='right')
    plt.savefig("data/plots/top_features.pdf")
    # plt.show()


def get_all_heuristic_models(g_test, nodes_test, g_train, nodes_train):
    nodes_pairs_test = [list(g_test.node_ids_to_ilocs([nodes_test[i][0], nodes_test[i][1]])) for i in
                        range(len(nodes_test))]
    nodes_pairs_train = [list(g_train.node_ids_to_ilocs([nodes_train[i][0], nodes_train[i][1]])) for i in
                         range(len(nodes_train))]

    adjacency = g_train.to_adjacency_matrix()
    list_features_test, list_features_train = [], []
    for method in [CommonNeighbors(), JaccardIndex(), SaltonIndex(), PreferentialAttachment(),
                   HubPromotedIndex(), HubDepressedIndex(), AdamicAdar(), ResourceAllocation()]:
        method.fit_predict(adjacency, 0)
        nodes_method_pairs_test = [method.predict((node1, node2)) for (node1, node2) in nodes_pairs_test]
        nodes_method_pairs_train = [method.predict((node1, node2)) for (node1, node2) in nodes_pairs_train]
        list_features_test.append(np.array(nodes_method_pairs_test)[:, np.newaxis])
        list_features_train.append(np.array(nodes_method_pairs_train)[:, np.newaxis])

    nodes_SP_pairs_test = [len(shortest_path(adjacency, node1, node2)) for (node1, node2) in nodes_pairs_test]
    nodes_SP_pairs_train = [len(shortest_path(adjacency, node1, node2)) for (node1, node2) in nodes_pairs_train]
    list_features_test.append(np.array(nodes_SP_pairs_test)[:, np.newaxis])
    list_features_train.append(np.array(nodes_SP_pairs_train)[:, np.newaxis])

    concat_feat_test = np.concatenate(list_features_test, axis=1)
    concat_feat_train = np.concatenate(list_features_train, axis=1)
    print(f"Heuristic concat_feat_test.shape: {concat_feat_test.shape}")
    print(f"Heuristic concat_feat_train.shape: {concat_feat_train.shape}")
    return concat_feat_test, concat_feat_train


def save_results_for_analysis(nodes_test, labels_test, predicted, method_name):
    with open('data/results/labels_test.npy', 'wb') as f:
        np.save(f, np.array(labels_test))
    with open('data/results/nodes_test.npy', 'wb') as f:
        np.save(f, np.array(nodes_test))
    with open(f'data/results/predicted_{method_name}.npy', 'wb') as f:
        np.save(f, np.array(predicted))


def SVM_all_features(all_txt_vis_embeddings, g_test, nodes_test, labels_test, g_train, nodes_train, labels_train):
    method_name = "_".join(["SVM", "all_txt_vis_embeddings_all_heuristics"])
    heuristic_concat_feat_test, heuristic_concat_feat_train = get_all_heuristic_models(g_test, nodes_test, g_train,
                                                                                       nodes_train)
    embedding_concat_feat_test, embedding_concat_feat_train = get_all_embedding_graphs(all_txt_vis_embeddings)

    print(f"Heuristic heuristic_concat_feat_train.shape: {heuristic_concat_feat_train.shape}")
    print(f"Embedding embedding_concat_feat_train.shape: {embedding_concat_feat_train.shape}")
    concat_feat_test = np.concatenate((embedding_concat_feat_test, heuristic_concat_feat_test), axis=1)
    concat_feat_train = np.concatenate((embedding_concat_feat_train, heuristic_concat_feat_train), axis=1)
    sc = StandardScaler()
    concat_feat_test = sc.fit_transform(concat_feat_test)
    concat_feat_train = sc.fit_transform(concat_feat_train)
    print(f"All concat_feat_train.shape: {concat_feat_train.shape}")

    method = svm.SVC()
    method.fit(concat_feat_train, labels_train)
    predicted = method.predict(concat_feat_test)
    accuracy = accuracy_score(labels_test, predicted) * 100

    console.print(f"Method {method_name}, max accuracy on test: {accuracy:.1f}", style="magenta")
    save_results_for_analysis(nodes_test, labels_test, predicted, method_name)  # optional, for error analysis


def SVM(g_test, nodes_test, labels_test, g_train, nodes_train, labels_train, feat_nodes):
    method_name = "_".join(["SVM", feat_nodes])

    nodes_feat_pairs_train = [g_train.node_features(nodes=[nodes_train[i][0], nodes_train[i][1]])
                              for i in range(len(nodes_train))]
    nodes_feat_pairs_test = [g_test.node_features(nodes=[nodes_test[i][0], nodes_test[i][1]])
                             for i in range(len(nodes_test))]

    nodes_feat_train = np.squeeze(
        np.array([nodes_feat_pairs.reshape(1, -1) for nodes_feat_pairs in nodes_feat_pairs_train]))
    nodes_feat_test = np.squeeze(
        np.array([nodes_feat_pairs.reshape(1, -1) for nodes_feat_pairs in nodes_feat_pairs_test]))

    method = svm.SVC()
    method.fit(nodes_feat_train, labels_train)
    predicted = method.predict(nodes_feat_test)
    accuracy = accuracy_score(labels_test, predicted) * 100
    console.print(f"Method {method_name}, max accuracy on test: {accuracy:.1f}", style="magenta")


def heuristic_methods(g_test, nodes_test, labels_test, g_train, g_val, nodes_val, labels_val):
    dict_method_threshold = finetune_threshold_on_validation(g_val, nodes_val, labels_val)
    print("Running heuristic methods on test...")
    for method_name in ["CommonNeighbors", "JaccardIndex", "SaltonIndex", "PreferentialAttachment", "AdamicAdar",
                        "HubPromotedIndex", "HubDepressedIndex", "ResourceAllocation", "ShortestPath"]:
        if not dict_method_threshold:
            threshold = 0.5
        else:
            threshold = dict_method_threshold[method_name]  # fine-tuned on validation
        if method_name == "CommonNeighbors":
            method = CommonNeighbors()
        elif method_name == "JaccardIndex":
            method = JaccardIndex()
        elif method_name == "SaltonIndex":
            method = SaltonIndex()
        elif method_name == "PreferentialAttachment":
            method = PreferentialAttachment()
        elif method_name == "HubPromotedIndex":
            method = HubPromotedIndex()
        elif method_name == "HubDepressedIndex":
            method = HubDepressedIndex()
        elif method_name == "AdamicAdar":
            method = AdamicAdar()
        elif method_name == "ResourceAllocation":
            method = ResourceAllocation()
        elif method_name == "ShortestPath":
            pass
        else:
            raise ValueError(f"method {method_name} nam not correct")

        console.print(f"Method {method_name}", style="magenta")

        adjacency = g_train.to_adjacency_matrix()
        if method_name != "ShortestPath":
            method.fit_predict(adjacency, 0)

        nodes_pairs_test = [list(g_test.node_ids_to_ilocs([nodes_test[i][0], nodes_test[i][1]])) for i in
                            range(len(nodes_test))]

        list_sim_predicted = []
        for (node1, node2), label in zip(nodes_pairs_test, labels_test):
            if method_name != "ShortestPath":
                common_neighbour_similarity = method.predict((node1, node2))
            else:
                list_shortest_path = shortest_path(adjacency, node1, node2)
                if not list_shortest_path:
                    common_neighbour_similarity = 0
                else:
                    common_neighbour_similarity = 1 / len(list_shortest_path)
            list_sim_predicted.append(common_neighbour_similarity)

        predicted = whitened_sigmoid(np.asarray(
            list_sim_predicted)) > threshold

        accuracy = accuracy_score(labels_test, predicted) * 100
        console.print(
            f"Accuracy on test {accuracy:.1f} with method {method_name} and fine-tuned threshold {threshold:.2f}",
            style="magenta")


def evaluate_GNN_model(model, model_name, train_flow, val_flow, test_flow):
    train_metrics = model.evaluate(train_flow)
    val_metrics = model.evaluate(val_flow)
    test_metrics = model.evaluate(test_flow)

    console.print(f"Train metrics of the trained model {model_name}:", style="magenta")
    for name, val in zip(model.metrics_names, train_metrics):
        console.print("\t{}: {:0.1f}".format(name, val * 100), style="magenta")

    console.print(f"Val metrics of the trained model {model_name}:", style="magenta")
    for name, val in zip(model.metrics_names, val_metrics):
        console.print("\t{}: {:0.1f}".format(name, val * 100), style="magenta")

    console.print(f"Test metrics of the trained model {model_name}:", style="magenta")
    for name, test in zip(model.metrics_names, test_metrics):
        console.print("\t{}: {:0.1f}".format(name, test * 100), style="magenta")
    print("-------------------------------------------------------")
    # write to file
    with open('data/utils/results.txt', 'a+') as results_file:
        file_console = Console(file=results_file)
        file_console.rule(f"Train metrics of the trained model {model_name}:", style="magenta")
        for name, val in zip(model.metrics_names, train_metrics):
            file_console.rule("\t{}: {:0.1f}".format(name, val * 100), style="magenta")

        file_console.rule(f"Val metrics of the trained model {model_name}:", style="magenta")
        for name, val in zip(model.metrics_names, val_metrics):
            file_console.rule("\t{}: {:0.1f}".format(name, val * 100), style="magenta")

        file_console.rule(f"Test metrics of the trained model {model_name}:", style="magenta")
        for name, test in zip(model.metrics_names, test_metrics):
            file_console.rule("\t{}: {:0.1f}".format(name, test * 100), style="magenta")
        file_console.rule("-------------------------------------------------------")


def GCN_model(g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test):
    train_gen = FullBatchLinkGenerator(g_train, method="gcn")
    val_gen = FullBatchLinkGenerator(g_val, method="gcn")
    test_gen = FullBatchLinkGenerator(g_test, method="gcn")

    gcn = GCN(
        layer_sizes=[16, 16], activations=["relu", "relu"], generator=train_gen, dropout=0.3
    )

    train_flow = train_gen.flow(nodes_train, labels_train)
    val_flow = val_gen.flow(nodes_val, labels_val)
    test_flow = test_gen.flow(nodes_test, labels_test)

    x_inp, x_out = gcn.in_out_tensors()

    '''
        Model
    '''
    # Final link classification layer that takes a pair of node embeddings produced by the GNN model,
    # applies a binary operator to them to produce the corresponding link embedding
    checkpoint_path = 'checkpoint/checkpoint_GCN/'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    binary_op = "dot"  # avg
    prediction = LinkEmbedding(activation="relu", method=binary_op)(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)
    callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=60)
    callback_mc = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_binary_accuracy',
                                                     save_weights_only=True, save_best_only=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.binary_crossentropy,
        metrics=["binary_accuracy"],
    )
    # Train model
    history = model.fit(
        train_flow, validation_data=val_flow, callbacks=[callback_es, callback_mc], epochs=300, verbose=1, shuffle=False
    )
    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(checkpoint_path)

    '''
        Get GNN node embeddings, after training
    '''
    # Get representations for all nodes in ``graph``
    embedding_model = keras.Model(inputs=x_inp, outputs=x_out)
    node_embeddings = embedding_model.predict(
        train_gen.flow(list(zip(g_train.nodes(), g_train.nodes())))
    )
    node_embeddings = node_embeddings[0][:, 0, :]

    with open('data/utils/GCN_node_embeddings.npy', 'wb') as f:
        np.save(f, node_embeddings)

    return train_flow, val_flow, test_flow, model


def GraphSage_model(g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test):
    batch_size = 20
    nb_epochs = 70
    layer_sizes = [50, 50]
    num_samples = [20, 10]

    train_gen = GraphSAGELinkGenerator(g_train, batch_size, num_samples)
    val_gen = GraphSAGELinkGenerator(g_val, batch_size, num_samples)
    test_gen = GraphSAGELinkGenerator(g_test, batch_size, num_samples)

    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.3, normalize="l2",
        activations=['relu', 'relu']
    )

    train_flow = train_gen.flow(nodes_train, labels_train, shuffle=True)
    val_flow = val_gen.flow(nodes_val, labels_val)
    test_flow = test_gen.flow(nodes_test, labels_test)

    x_inp, x_out = graphsage.in_out_tensors()

    '''
        Model
    '''
    checkpoint_path = 'checkpoint/checkpoint_GraphSage/'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    prediction = LinkEmbedding(activation="relu", method="dot")(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)
    callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=30)
    callback_mc = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_binary_accuracy',
                                                     save_weights_only=True, save_best_only=True)
    # callback_tf = tf.keras.callbacks.TensorBoard(log_dir='./logs') #run: tensorboard --logdir=./logs --bind_all
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.binary_crossentropy,
        metrics=["binary_accuracy"],
    )
    # print(model.summary())

    # Train model
    history = model.fit(
        train_flow, validation_data=val_flow, callbacks=[callback_es, callback_mc], epochs=nb_epochs, verbose=1
    )
    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(checkpoint_path)

    '''
        Get GNN node embeddings, after training
    '''
    # Build the model to predict node representations from node features with the learned GraphSAGE model parameters
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    # Get representations for all nodes in ``graph``
    node_gen = GraphSAGENodeGenerator(g_train, batch_size, num_samples).flow(
        g_train.nodes()
    )
    node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=0)

    with open('data/utils/graphsage_node_embeddings.npy', 'wb') as f:
        np.save(f, node_embeddings)

    return train_flow, val_flow, test_flow, model


def attri2vec_model(g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test):
    batch_size = 50
    nb_epochs = 200
    layer_sizes = [128]

    train_gen = Attri2VecLinkGenerator(g_train, batch_size)
    val_gen = Attri2VecLinkGenerator(g_val, batch_size)
    test_gen = Attri2VecLinkGenerator(g_test, batch_size)

    attri2vec = Attri2Vec(
        layer_sizes=layer_sizes, generator=train_gen, activation='relu', bias=False, normalize='l2'
    )

    train_flow = train_gen.flow(nodes_train, labels_train)
    val_flow = val_gen.flow(nodes_val, labels_val)
    test_flow = test_gen.flow(nodes_test, labels_test)

    x_inp, x_out = attri2vec.in_out_tensors()
    '''
        Model
    '''
    checkpoint_path = 'checkpoint/checkpoint_Attri2vec/'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    prediction = LinkEmbedding(activation="relu", method="dot")(x_out)
    # prediction = Dropout(0.1)(prediction) # Attri2vec is overfitting - not working
    model = keras.Model(inputs=x_inp, outputs=prediction)
    callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=50)
    callback_mc = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_binary_accuracy',
                                                     save_weights_only=True, save_best_only=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.binary_crossentropy,
        metrics=["binary_accuracy"],
    )

    # Train model
    history = model.fit(
        train_flow, validation_data=val_flow, callbacks=[callback_es, callback_mc], epochs=nb_epochs, verbose=1,
        shuffle=False
    )
    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(checkpoint_path)
    '''
        Get GNN node embeddings, after training
    '''
    # Build the model to predict node representations from node features with the learned Attri2Vec model parameters
    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    # Get representations for all nodes in ``graph``
    node_gen = Attri2VecNodeGenerator(g_train, batch_size).flow(g_train.nodes())
    node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=0)

    with open('data/utils/attri2vec_node_embeddings.npy', 'wb') as f:
        np.save(f, node_embeddings)

    return train_flow, val_flow, test_flow, model


def node2vec_model(g_train, g_val, g_test, nodes_val, nodes_test, labels_val, labels_test):
    walk_number = 30  # 100
    walk_length = 10  # 5  # Larger values can be set to them to achieve better performance
    batch_size = 50
    emb_size = 128
    nb_epochs = 60  # 10

    # Create the biased random walker to generate random walks
    walker = create_biased_random_walker(g_train, walk_number, walk_length)

    # Create the UnsupervisedSampler instance with the biased random walker
    unsupervised_samples = UnsupervisedSampler(g_train, nodes=list(g_train.nodes()), walker=walker)

    train_gen = Node2VecLinkGenerator(g_train, batch_size)
    val_gen = Node2VecLinkGenerator(g_val, batch_size)
    test_gen = Node2VecLinkGenerator(g_test, batch_size)

    node2vec = Node2Vec(emb_size, generator=train_gen)

    train_flow = train_gen.flow(unsupervised_samples)
    val_flow = val_gen.flow(nodes_val, labels_val)
    test_flow = test_gen.flow(nodes_test, labels_test)

    x_inp, x_out = node2vec.in_out_tensors()

    '''
        Model
    '''
    checkpoint_path: str = 'checkpoint/checkpoint_Node2vec/'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    prediction = LinkEmbedding(activation="relu", method="dot")(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)
    callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=20)
    callback_mc = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_binary_accuracy',
                                                     save_weights_only=True, save_best_only=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.binary_crossentropy,
        metrics=["binary_accuracy"],
    )

    # Train model
    history = model.fit(
        train_flow, validation_data=val_flow, callbacks=[callback_es, callback_mc], epochs=nb_epochs, verbose=0,
        shuffle=False
    )
    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(checkpoint_path)
    '''
        Get GNN node embeddings, after training
    '''
    # Build the model to predict node representations from node ids with the learned Node2Vec model parameters
    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    # Get representations for all nodes in ``graph``
    node_gen = Node2VecNodeGenerator(g_train, batch_size).flow(g_train.nodes())
    node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=0)

    with open('data/utils/node2vec_node_embeddings.npy', 'wb') as f:
        np.save(f, node_embeddings)

    return train_flow, val_flow, test_flow, model


def GNN_methods(g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test):
    for model_name in ["GCN", "GraphSage", "Attri2vec", "Node2vec"]:
        if model_name == "GCN":
            train_flow, val_flow, test_flow, model = GCN_model(g_train, g_val, g_test,
                                                               nodes_train, nodes_val,
                                                               nodes_test, labels_train,
                                                               labels_val, labels_test)
        elif model_name == "GraphSage":
            train_flow, val_flow, test_flow, model = GraphSage_model(g_train, g_val, g_test,
                                                                     nodes_train, nodes_val,
                                                                     nodes_test, labels_train,
                                                                     labels_val, labels_test)
        elif model_name == "Attri2vec":
            train_flow, val_flow, test_flow, model = attri2vec_model(g_train, g_val, g_test,
                                                                     nodes_train, nodes_val,
                                                                     nodes_test, labels_train,
                                                                     labels_val, labels_test)
        elif model_name == "Node2vec":
            train_flow, val_flow, test_flow, model = node2vec_model(g_train, g_val, g_test,
                                                                    nodes_val, nodes_test,
                                                                    labels_val, labels_test)
        else:
            raise ValueError("Wrong graph embeddings name!!")
        evaluate_GNN_model(model, model_name, train_flow, val_flow, test_flow)


def get_nearest_neighbours():
    # SentenceBert embeddings
    txt_action_node_data = pd.read_csv('data/graph/txt_action_nodes.csv', index_col=0)
    txt_action_node_data_values = txt_action_node_data.values
    txt_action_node_data_names = txt_action_node_data.index.values.tolist()

    # CLIP vis embeddings
    vis_video_node_data = pd.read_csv('data/graph/vis_video_nodes.csv', index_col=0)
    vis_video_node_data_values = vis_video_node_data.values
    vis_video_node_data_names = vis_video_node_data.index.values.tolist()

    # AverageNeighbourWeight embeddings
    avg_node_data = pd.read_csv('data/graph/graph_txt_action_nodes.csv', index_col=0)
    avg_node_data_values = avg_node_data.values
    avg_node_data_names = avg_node_data.index.values.tolist()

    list_check_actions = ["add tea", "build desk", "squeeze lemon juice", "rub stain", "chop potato"]
    top_k = 10
    for check_action in list_check_actions:
        knn = NearestNeighbors(n_neighbors=top_k)
        knn.fit(txt_action_node_data_values)
        index_action = txt_action_node_data_names.index(check_action)
        check_action_features = txt_action_node_data_values[index_action].reshape(1, -1)
        list_indexes = knn.kneighbors(check_action_features, return_distance=False)
        neighbours = [txt_action_node_data_names[index] for index in list_indexes[0]]
        console.print(f"SBert Neighbours for: {check_action}: {neighbours}", style="magenta")

        knn = NearestNeighbors(n_neighbors=top_k)
        knn.fit(vis_video_node_data_values)
        index_action = vis_video_node_data_names.index(check_action)
        check_action_features = vis_video_node_data_values[index_action].reshape(1, -1)
        list_indexes = knn.kneighbors(check_action_features, return_distance=False)
        neighbours = [vis_video_node_data_names[index] for index in list_indexes[0]]
        console.print(f"VisCLIP Neighbours for: {check_action}: {neighbours}", style="magenta")

        knn = NearestNeighbors(n_neighbors=top_k)
        knn.fit(avg_node_data_values)
        index_action = avg_node_data_names.index(check_action)
        check_action_features = avg_node_data_values[index_action].reshape(1, -1)
        list_indexes = knn.kneighbors(check_action_features, return_distance=False)
        neighbours = [avg_node_data_names[index] for index in list_indexes[0]]
        console.print(f"Graph Neighbours for: {check_action}: {neighbours}", style="magenta")


def finetune_threshold_cosine_similarity(g_val, nodes_val, labels_val, method_name):
    nodes_feat_pairs = [g_val.node_features(nodes=[nodes_val[i][0], nodes_val[i][1]])
                        for i in range(len(nodes_val))]
    list_sim_predicted = [cosine_similarity(node1_feat.reshape(1, -1), node2_feat.reshape(1, -1))[0][0] for
                          [node1_feat, node2_feat] in nodes_feat_pairs]

    max_accuracy = 0
    max_threshold = 0
    for threshold in np.linspace(-1, 1, 10).tolist():
        # predicted = whitened_sigmoid(np.asarray(list_sim_predicted)) > threshold  #keep whitened sigmoid?
        predicted = np.asarray(list_sim_predicted) > threshold
        accuracy = accuracy_score(labels_val, predicted)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_threshold = threshold
    max_accuracy = max_accuracy * 100
    console.print(
        f"Method {method_name}, on validation max accuracy: {max_accuracy:.1f} with threshold: {max_threshold:.2f}",
        style="magenta")
    return max_threshold


def similarity_method(g_test, nodes_test, labels_test, g_val, nodes_val, labels_val, feat_nodes):
    method_name = "_".join(["similarity", feat_nodes])

    nodes_feat_pairs = [g_test.node_features(nodes=[nodes_test[i][0], nodes_test[i][1]]) for i in
                        range(len(nodes_test))]
    list_sim_predicted = [cosine_similarity(node1_feat.reshape(1, -1), node2_feat.reshape(1, -1))[0][0] for
                          [node1_feat, node2_feat] in nodes_feat_pairs]

    threshold = finetune_threshold_cosine_similarity(g_val, nodes_val, labels_val, method_name)

    predicted = np.asarray(list_sim_predicted) > threshold
    accuracy = accuracy_score(labels_test, predicted) * 100

    console.print(f"Method {method_name}, max accuracy on test: {accuracy:.1f} with threshold: {threshold:.2f}",
                  style="magenta")


def save_graph_embeddings(feat_nodes):
    g = test_my_data(input_nodes='data/graph/' + feat_nodes + '_nodes.csv',
                     input_edges='data/graph/edges.csv')
    g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test = \
        test_val_train_split(g)

    list_weighted_avg_embeddings = []
    self_weight = 1
    for node in track(g_train.nodes(), description="Computing graph embeddings from weighted avg of neighbours..."):
        node_emb_weighted = g_train.node_features(nodes=[node]) * self_weight
        sum_weights = self_weight
        for node_neighbour, edge_weight in g_train.in_nodes(node, include_edge_weight=True):
            edge_weight = 1
            node_emb_weighted += g_train.node_features(nodes=[node_neighbour]) * edge_weight
            sum_weights += edge_weight
        list_weighted_avg_embeddings.append(node_emb_weighted / sum_weights)  # weighted edge mean of neighbour nodes

    df = pd.DataFrame([np.squeeze(tensor) for tensor in list_weighted_avg_embeddings], index=list(g_train.nodes()))
    df.to_csv('data/graph/graph_' + feat_nodes + '_nodes.csv')


def get_all_embedding_graphs(all_txt_vis_embeddings):
    list_features_train, list_features_test = [], []
    for feat_nodes in all_txt_vis_embeddings:
        g = test_my_data(input_nodes=f'data/graph/{feat_nodes}_nodes.csv',
                         input_edges='data/graph/edges.csv')  # one_edges
        g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test = \
            test_val_train_split(g)

        nodes_feat_pairs_test = [g_val.node_features(nodes=[nodes_test[i][0], nodes_test[i][1]])
                                 for i in range(len(nodes_test))]
        nodes_feat_pairs_train = [g_train.node_features(nodes=[nodes_train[i][0], nodes_train[i][1]])
                                  for i in range(len(nodes_train))]

        nodes_feat_test = np.squeeze(np.array([nodes_feat_pairs.reshape(1, -1)
                                               for nodes_feat_pairs in nodes_feat_pairs_test]))
        nodes_feat_train = np.squeeze(np.array([nodes_feat_pairs.reshape(1, -1)
                                                for nodes_feat_pairs in nodes_feat_pairs_train]))

        list_features_test.append(nodes_feat_test)
        list_features_train.append(nodes_feat_train)
    concat_feat_train = np.concatenate(list_features_train, axis=1)
    concat_feat_test = np.concatenate(list_features_test, axis=1)

    # sc = StandardScaler()
    # # pca = PCA(n_components=2)
    # nodes_feat_train = sc.fit_transform(concat_feat_train)
    # concat_feat_test = sc.transform(concat_feat_test)
    # # pca.fit_transform(nodes_feat_train)
    # # nodes_feat_test = pca.transform(nodes_feat_test)

    print(f"Embedding concat_feat_test.shape: {concat_feat_test.shape}")
    print(f"Embedding concat_feat_train.shape: {concat_feat_train.shape}")
    return concat_feat_test, concat_feat_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--compute_graph_embeddings', action='store_true')
    parser.add_argument('-r', '--run_method', action='store_true')
    parser.add_argument('-m', '--method', choices=["heuristic", "GNN", "cosine", "SVM"])
    parser.add_argument('-nn', '--get_nearest_neighbours', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_txt_vis_embeddings = ["txt_action", "txt_transcript", "vis_action", "vis_video", "vis_action_video"]
    all_graph_embeddings = ["graph_" + embedding for embedding in all_txt_vis_embeddings]

    if args.compute_graph_embeddings:
        for feat_nodes in all_txt_vis_embeddings:
            save_graph_embeddings(feat_nodes)

    if args.run_method:
        default_feat_nodes = "txt_action"
        g = test_my_data(input_nodes=f'data/graph/{default_feat_nodes}_nodes.csv',
                         input_edges='data/graph/edges.csv')
        g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test = \
            test_val_train_split(g)

        if args.method == "heuristic":
            heuristic_methods(g_test, nodes_test, labels_test, g_train, g_val, nodes_val, labels_val)
            weighted_heuristic_methods(nodes_test, labels_test, g_val, nodes_val, labels_val)

        elif args.method == "GNN":
            GNN_methods(g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test)

        elif args.method == "cosine":
            # ablation per input representation/ embedding type
            for feat_nodes in all_txt_vis_embeddings + all_graph_embeddings:
                g = test_my_data(input_nodes=f'data/graph/{feat_nodes}_nodes.csv',
                                 input_edges='data/graph/edges.csv')
                g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test = \
                    test_val_train_split(g)

                similarity_method(g_test, nodes_test, labels_test, g_val, nodes_val, labels_val, feat_nodes)

        elif args.method == "SVM":
            # ablation per input representation/ embedding type
            for feat_nodes in all_txt_vis_embeddings + all_graph_embeddings:
                g = test_my_data(input_nodes=f'data/graph/{feat_nodes}_nodes.csv',
                                 input_edges='data/graph/edges.csv')
                g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test = \
                    test_val_train_split(g)
                SVM(g_test, nodes_test, labels_test, g_train, nodes_train, labels_train, feat_nodes)

            SVM_all_features(all_txt_vis_embeddings + all_graph_embeddings, g_test, nodes_test, labels_test, g_train,
                             nodes_train, labels_train)
        else:
            raise ValueError(f"Unknown method: {args.method}")

    if args.get_nearest_neighbours:
        get_nearest_neighbours()


if __name__ == '__main__':
    main()

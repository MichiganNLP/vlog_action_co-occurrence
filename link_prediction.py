import json
import os
from collections import defaultdict, Counter

import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display, HTML
from rich.console import Console
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sknetwork.linkpred import CommonNeighbors, JaccardIndex, SaltonIndex, HubPromotedIndex, AdamicAdar, \
    ResourceAllocation, PreferentialAttachment, HubDepressedIndex, whitened_sigmoid
from sknetwork.path import shortest_path
from stellargraph import StellarGraph
from stellargraph import datasets
from stellargraph.data import EdgeSplitter, BiasedRandomWalk, UnsupervisedSampler
from stellargraph.layer import GCN, LinkEmbedding, Node2Vec, Attri2Vec, GraphSAGE
from stellargraph.mapper import FullBatchLinkGenerator, Node2VecLinkGenerator, Node2VecNodeGenerator, \
    Attri2VecLinkGenerator, Attri2VecNodeGenerator, GraphSAGELinkGenerator, GraphSAGENodeGenerator
from tensorflow import keras
from rich.progress import track

from data_analysis import get_all_actions, get_all_action_pairs

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


def test_cora():
    dataset = datasets.Cora()
    display(HTML(dataset.description))
    g, _ = dataset.load(subject_as_feature=True)
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
    edge_splitter_train = EdgeSplitter(g_test)

    g_train, nodes, labels = edge_splitter_train.train_test_split(
        p=0.1, method="global"
    )

    (
        nodes_train,
        nodes_val,  # validation, model_selection
        labels_train,
        labels_val,
    ) = train_test_split(nodes, labels, train_size=0.75, test_size=0.25)

    print(g_train.info())
    print(len(nodes_test), len(nodes_train), len(nodes_val))
    return g_train, g_test, nodes_train, nodes_val, labels_train, labels_val, nodes_test, labels_test


def test_val_train_split2(g):
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


def test_train_split(g):
    # Define an edge splitter on the original graph g:
    edge_splitter_test = EdgeSplitter(g)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from g, and obtain the
    # reduced graph g_test with the sampled links removed:
    g_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global", keep_connected=True, seed=random_seed
    )

    # Define an edge splitter on the reduced graph g_test:
    edge_splitter_train = EdgeSplitter(g_test)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from g_test, and obtain the
    # reduced graph g_train with the sampled links removed:
    g_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=0.1, method="global", keep_connected=True, seed=random_seed
    )

    return g_train, g_test, edge_ids_train, edge_labels_train, edge_ids_test, edge_labels_test


def weighted_heuristic_methods(g_val, edge_ids_val, edge_labels_val):
    nodes_names = [list([edge_ids_val[i][0], edge_ids_val[i][1]]) for i in range(len(edge_ids_val))]
    # nodes_val = [list(g_val.node_ids_to_ilocs([edge_ids_val[i][0], edge_ids_val[i][1]])) for i in
    #              range(len(edge_ids_val))]

    all_edge_info = g_val.edge_arrays(include_edge_weight=True)
    source, target, weights = all_edge_info[0].tolist(), all_edge_info[1].tolist(), all_edge_info[3].tolist()

    dict_common_neighbours = defaultdict()
    dict_common_neighbours_w = defaultdict()
    for (node1, node2) in track(nodes_names):
        common_neighbours = set(g_val.in_nodes(node1)).intersection(set(g_val.in_nodes(node2)))
        dict_common_neighbours[(node1, node2)] = common_neighbours
        total_weight = 0
        for n in common_neighbours:
            for (s, t, w) in zip(source, target, weights):
                if (node1 == s and n == t) or (node2 == s and n == t) or (node1 == t and n == s) or (node2 == t and n == s):
                    total_weight += w
        dict_common_neighbours_w[(node1, node2)] = total_weight

    print(dict_common_neighbours_w)
    # print(node1, node2, label, common_neighbours)
    # dict_info = defaultdict()

    # for (node1, node2) in dict_common_neighbours:
    #     for (s, t, w) in zip(source, target, weights):
    #         if (s == source and t == target) or (s == target and t == source):
    #             dict_info[(s, t)].append()



def finetune_threshold_on_validation(g_val, edge_ids_val, edge_labels_val, dict_method_threshold):
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

        nodes_val = [list(g_val.node_ids_to_ilocs([edge_ids_val[i][0], edge_ids_val[i][1]])) for i in
                     range(len(edge_ids_val))]

        list_sim_predicted = []
        for (node1, node2), label in zip(nodes_val, edge_labels_val):
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
            predicted = whitened_sigmoid(np.asarray(list_sim_predicted)) > threshold  # TODO: keep whitened sigmoid?
            # predicted = np.asarray(list_sym_predicted) > threshold  # TODO: keep whitened sigmoid?
            accuracy = accuracy_score(edge_labels_val, predicted)
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


def SVM(g_test, edge_ids_test, edge_labels_test, g_val, method_name):
    nodes_feat_pairs_test = [g_val.node_features(nodes=[edge_ids_test[i][0], edge_ids_test[i][1]])
                             for i in range(len(edge_ids_test))]

    adjacency = g_val.to_adjacency_matrix()
    # method_HD = HubDepressedIndex()
    # method_HD.fit_predict(adjacency, 0)

    nodes_pairs_test = [list(g_test.node_ids_to_ilocs([edge_ids_test[i][0], edge_ids_test[i][1]])) for i in
                        range(len(edge_ids_test))]
    nodes_SP_pairs_test = [len(shortest_path(adjacency, node1, node2)) for (node1, node2) in nodes_pairs_test]
    # nodes_HD_pairs_test = [method_HD.predict((node1, node2)) for (node1, node2) in nodes_pairs_test]

    nodes_feat1_test = np.squeeze(
        np.array([nodes_feat_pairs_feat.reshape(1, -1) for nodes_feat_pairs_feat in nodes_feat_pairs_test]))
    nodes_feat2_test = np.array(nodes_SP_pairs_test)[:, np.newaxis]
    # nodes_feat3_test = np.array(nodes_HD_pairs_test)[:, np.newaxis]
    concat_feat_test = nodes_feat1_test
    # concat_feat_test = np.concatenate((nodes_feat1_test, nodes_feat2_test), axis=1)


    from sklearn import svm
    method = svm.SVC()
    method.fit(concat_feat_test, edge_labels_test)
    predicted = method.predict(concat_feat_test)
    accuracy = accuracy_score(edge_labels_test, predicted) * 100
    # coef = method.coef_.ravel()
    # print(coef)
    # print(Counter(predicted))
    # print(Counter(edge_labels_test))

    console.print(f"Method {method_name}, max accuracy on test: {accuracy:.1f}", style="magenta")


def heuristic_methods(g_test, edge_ids_test, edge_labels_test, g_val, dict_method_threshold):
    print("Running heuristic methods on test...")
    # TODO: Katz, Leich-Holme-Newman?

    for method_name in ["CommonNeighbors", "JaccardIndex", "SaltonIndex", "PreferentialAttachment", "AdamicAdar",
                        "HubPromotedIndex", "HubDepressedIndex", "ResourceAllocation", "ShortestPath"]:
        if not dict_method_threshold:
            threshold = 0.5
        else:
            threshold = dict_method_threshold[method_name]  # finetuned on validation

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

        adjacency = g_val.to_adjacency_matrix()
        if method_name != "ShortestPath":
            method.fit_predict(adjacency, 0)  # assigns a scores to edges
            # method.predict(list(range(adjacency.shape[0])))

        nodes_test = [list(g_test.node_ids_to_ilocs([edge_ids_test[i][0], edge_ids_test[i][1]])) for i in
                      range(len(edge_ids_test))]

        # predicted = []
        list_sim_predicted = []
        for (node1, node2), label in zip(nodes_test, edge_labels_test):
            if method_name != "ShortestPath":
                common_neighbour_similarity = method.predict((node1, node2))
            else:
                list_shortest_path = shortest_path(adjacency, node1, node2)
                if not list_shortest_path:
                    common_neighbour_similarity = 0
                else:
                    common_neighbour_similarity = 1 / len(list_shortest_path)
            list_sim_predicted.append(common_neighbour_similarity)
            # link = 0 if common_neighbour_similarity < threshold else 1
            # predicted.append(link)

        predicted = whitened_sigmoid(np.asarray(
            list_sim_predicted)) > threshold  # TODO: keep whitened sigmoid? https://scikit-network.readthedocs.io/en/latest/tutorials/linkpred/first_order.html

        # print(f"Test GT data labels: {Counter(edge_labels_test)}")
        # print(f"Test Pred data sym scores: {Counter(list_sym_predicted)}")
        # print(f"Test Pred data labels: {Counter(predicted)}")

        accuracy = accuracy_score(edge_labels_test, predicted) * 100
        console.print(
            f"Accuracy on test {accuracy:.1f} with method {method_name} and fine-tuned threshold {threshold:.2f}",
            style="magenta")


def evaluate_GNN_model(model, history, model_name, train_flow, val_flow, test_flow):
    # fig = sg.utils.plot_history(history, return_figure=True)
    # fig.savefig(f'data/plots/history_{model_name}_loss.pdf')

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
    # print(model.summary())
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

    return train_flow, val_flow, test_flow, history, model


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

    # def get_embedding(u):
    #     u_index = g_train.nodes().index(u)
    #     return node_embeddings[u_index]

    with open('data/utils/graphsage_node_embeddings.npy', 'wb') as f:
        np.save(f, node_embeddings)

    return train_flow, val_flow, test_flow, history, model


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
    # print(model.summary())

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

    return train_flow, val_flow, test_flow, history, model


def node2vec_model(g_train, g_val, g_test, nodes_val, nodes_test, labels_val, labels_test):
    walk_number = 30  # 100
    walk_length = 10  # 5  # Larger values can be set to them to achieve better performance
    batch_size = 50
    emb_size = 128
    nb_epochs = 60  # 10

    # TODO
    # num_walks = 10
    # walk_length = 80

    # walker = BiasedRandomWalk(
    #     g_train,
    #     n=walk_number,
    #     length=walk_length,
    #     p=0.5,  # defines probability, 1/p, of returning to source node
    #     q=2.0,  # defines probability, 1/q, for moving to a node away from the source node
    # )

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
    # print(model.summary())

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

    return train_flow, val_flow, test_flow, history, model


def GNN_link_model(g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test,
                   feat_nodes):
    # for model_name in ["GraphSage", "Attri2vec", "Node2vec"]: #GCN
    for model_name in ["GCN"]:  # GCN
        if model_name == "GCN":
            train_flow, val_flow, test_flow, history, model = GCN_model(g_train, g_val, g_test,
                                                                        nodes_train, nodes_val,
                                                                        nodes_test, labels_train,
                                                                        labels_val, labels_test)
        elif model_name == "GraphSage":
            train_flow, val_flow, test_flow, history, model = GraphSage_model(g_train, g_val, g_test,
                                                                              nodes_train, nodes_val,
                                                                              nodes_test, labels_train,
                                                                              labels_val, labels_test)
        elif model_name == "Attri2vec":
            train_flow, val_flow, test_flow, history, model = attri2vec_model(g_train, g_val, g_test,
                                                                              nodes_train, nodes_val,
                                                                              nodes_test, labels_train,
                                                                              labels_val, labels_test)
        elif model_name == "Node2vec":
            train_flow, val_flow, test_flow, history, model = node2vec_model(g_train, g_val, g_test,
                                                                             nodes_val, nodes_test,
                                                                             labels_val, labels_test)
        else:
            raise ValueError("Wrong graph embeddings name!!")
        # model_name = "Node2vec"
        # for (walk_number, walk_length) in [(30, 30)]:
        #     console.print(f"walk nb: {walk_number}, walk length: {walk_length}", style="magenta")
        #     train_flow, val_flow, test_flow, history, model = node2vec_model(g_train, g_val, g_test,
        #                                                                      nodes_val, nodes_test,
        #                                                                      labels_val, labels_test, walk_number, walk_length)
        #     evaluate_GNN_model(model, history, model_name, train_flow, val_flow, test_flow)
        model_name = "_".join([model_name, feat_nodes])
        evaluate_GNN_model(model, history, model_name, train_flow, val_flow, test_flow)


def get_nearest_neighbours():
    # SentenceBert embeddings
    stsbrt_node_data = pd.read_csv('data/graph/all_stsbrt_nodes.csv', index_col=0)
    stsbrt_node_data_values = stsbrt_node_data.values
    stsbrt_node_data_names = stsbrt_node_data.index.values.tolist()

    # AverageNeighbourWeight embeddings
    avg_node_data = pd.read_csv('data/graph/all_weighted_avg_nodes.csv', index_col=0)
    avg_node_data_values = avg_node_data.values
    avg_node_data_names = avg_node_data.index.values.tolist()

    # for method_name in ["GCN", "graphsage", "attri2vec", "node2vec"]:
    # for method_name in ["GCN"]:
    #     with open('data/utils/' + method_name + '_node_embeddings.npy', 'rb') as f:
    #         gnn_node_data_values = np.load(f)

    list_check_actions = ["add tea", "build desk", "squeeze lemon juice", "rub stain", "chop potato"]
    # list_check_actions = ["raise baby"]
    top_k = 10
    for check_action in list_check_actions:
        knn = NearestNeighbors(n_neighbors=top_k)
        knn.fit(stsbrt_node_data_values)
        index_action = stsbrt_node_data_names.index(check_action)
        check_action_features = stsbrt_node_data_values[index_action].reshape(1, -1)
        list_indexes = knn.kneighbors(check_action_features, return_distance=False)
        neighbours = [stsbrt_node_data_names[index] for index in list_indexes[0]]
        console.print(f"SentenceBert Neighbours for: {check_action}: {neighbours}", style="magenta")

        knn = NearestNeighbors(n_neighbors=top_k)
        knn.fit(avg_node_data_values)
        index_action = avg_node_data_names.index(check_action)
        check_action_features = avg_node_data_values[index_action].reshape(1, -1)
        list_indexes = knn.kneighbors(check_action_features, return_distance=False)
        neighbours = [avg_node_data_names[index] for index in list_indexes[0]]
        console.print(f"AverageWeight Neighbours for: {check_action}: {neighbours}", style="magenta")

        # knn = NearestNeighbors(n_neighbors=6)
        # knn.fit(gnn_node_data_values)
        # index_action = list(g.nodes()).index(check_action)
        # check_action_features = gnn_node_data_values[index_action].reshape(1, -1)
        # list_indexes = knn.kneighbors(check_action_features, return_distance=False)
        # neighbours = [stsbrt_node_data_names[index] for index in list_indexes[0]]
        # console.print(f"{method_name} Neighbours for: {check_action}: {neighbours}", style="magenta")

        # index_action = list(g.nodes()).index('use serenity')
        # action_emb1 = gnn_node_data_values[index_action].reshape(1, -1)
        #
        # index_action = list(g.nodes()).index('raise baby')
        # action_emb2 = gnn_node_data_values[index_action].reshape(1, -1)
        #
        # index_action = list(g.nodes()).index('read book')
        # action_emb3 = gnn_node_data_values[index_action].reshape(1, -1)
        # print(cosine_similarity(action_emb1, action_emb2), cosine_similarity(action_emb2, action_emb3))


def finetune_threshold_similarity_method(g_val, edge_ids_val, edge_labels_val, method_name):
    nodes_feat_pairs = [g_val.node_features(nodes=[edge_ids_val[i][0], edge_ids_val[i][1]])
                        for i in range(len(edge_ids_val))]
    list_sim_predicted = [cosine_similarity(node1_feat.reshape(1, -1), node2_feat.reshape(1, -1))[0][0] for
                          [node1_feat, node2_feat] in nodes_feat_pairs]

    max_accuracy = 0
    max_threshold = 0
    for threshold in np.linspace(-1, 1, 10).tolist():
        # predicted = whitened_sigmoid(np.asarray(list_sym_predicted)) > threshold  # TODO: keep whitened sigmoid?
        predicted = np.asarray(list_sim_predicted) > threshold  # TODO: keep whitened sigmoid?
        accuracy = accuracy_score(edge_labels_val, predicted)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_threshold = threshold
    max_accuracy = max_accuracy * 100
    console.print(
        f"Method {method_name}, on validation max accuracy: {max_accuracy:.1f} with threshold: {max_threshold:.2f}",
        style="magenta")
    return max_threshold


# baseline no graph method
def similarity_method(g_test, edge_ids_test, edge_labels_test, g_val, edge_ids_val, edge_labels_val, method_name):
    # nodes_name_pairs = [[edge_ids_val[i][0], edge_ids_val[i][1]] for i in range(len(edge_ids_val))]
    # nodes_val = [list(g_val.node_ids_to_ilocs([edge_ids_val[i][0], edge_ids_val[i][1]])) for i in range(len(edge_ids_val))]

    nodes_feat_pairs = [g_test.node_features(nodes=[edge_ids_test[i][0], edge_ids_test[i][1]])
                        for i in range(len(edge_ids_test))]
    list_sim_predicted = [cosine_similarity(node1_feat.reshape(1, -1), node2_feat.reshape(1, -1))[0][0] for
                          [node1_feat, node2_feat] in nodes_feat_pairs]

    threshold = finetune_threshold_similarity_method(g_val, edge_ids_val, edge_labels_val, method_name)

    predicted = np.asarray(list_sim_predicted) > threshold
    accuracy = accuracy_score(edge_labels_test, predicted) * 100

    console.print(f"Method {method_name}, max accuracy on test: {accuracy:.1f} with threshold: {threshold:.2f}",
                  style="magenta")
    print(Counter(predicted))
    print(Counter(edge_labels_test))

    return predicted, edge_labels_test, edge_ids_test


def get_graph_weighted_embeddings(feat_nodes):
    g = test_my_data(input_nodes='data/graph/all_' + feat_nodes + '_nodes.csv',
                     input_edges='data/graph/all_edges_missing.csv')
    g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test = \
        test_val_train_split2(g)

    list_weighted_avg_embeddings = []
    self_weight = 1
    for node in track(g_val.nodes(), description="Computing graph embeddings from weighted avg of neighbours..."):
        node_emb_weighted = g_val.node_features(nodes=[node]) * self_weight
        sum_weights = self_weight
        for node_neighbour, edge_weight in g_val.in_nodes(node, include_edge_weight=True):
            node_emb_weighted += g_val.node_features(nodes=[node_neighbour]) * edge_weight
            sum_weights += edge_weight
        list_weighted_avg_embeddings.append(node_emb_weighted / sum_weights)  # weighted edge mean of neighbour nodes

    df = pd.DataFrame([np.squeeze(tensor) for tensor in list_weighted_avg_embeddings], index=list(g_val.nodes()))
    df.to_csv('data/graph/all_weighted_' + feat_nodes + '_nodes.csv')


def analyse_results(predicted_labels, edge_labels_test, edges_ids_test):
    with open('data/dict_video_action_pairs_filtered_by_link.json') as json_file:
        dict_video_action_pairs_filtered = json.load(json_file)

    all_action_pairs = get_all_action_pairs(dict_video_action_pairs_filtered)
    all_actions = get_all_actions(all_action_pairs)

    nodes_name_pairs = [(edges_ids_test[i][0], edges_ids_test[i][1]) for i in range(len(edges_ids_test))]

    dict_results_correct = defaultdict()
    dict_results_errors = {"FP": {}, "FN": {}}
    for i, (predicted, gt) in enumerate(zip(predicted_labels, edge_labels_test)):
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
    # g = test_cora()
    dict_method_threshold = {}
    all_embeddings = ["stsbrt", "transcript_stsbrt", "txtclip", "visclip", "avgclip"]
    all_weighted_embeddings = ["weighted_" + el for el in all_embeddings]

    # for feat_nodes in all_weighted_embeddings:
    # for feat_nodes in all_embeddings:
    for feat_nodes in ["stsbrt"]:
        # get_graph_weighted_embeddings(feat_nodes)

        g = test_my_data(input_nodes=f'data/graph/all_{feat_nodes}_nodes.csv',
                         # input_edges='data/graph/all_edges.csv')
                         input_edges='data/graph/all_edges_missing.csv')
        # input_edges='data/graph/all_one_edges.csv')

        g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test = \
            test_val_train_split2(g)

        # SVM(g_test, nodes_test, labels_test, g_val, method_name="_".join(["SVM", feat_nodes])) #TODO: PIPELINE select from all features

        # predicted, edge_labels_test, edges_ids_test = similarity_method(g_test, nodes_test, labels_test, g_val,
        #                                                                 nodes_val, labels_val,
        #                                                                 method_name="_".join(
        #                                                                     ["Similarity", feat_nodes]))

        # analyse_results(predicted, edge_labels_test, edges_ids_test)

        # GNN_link_model(g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train,
        #                labels_val,
        #                labels_test, feat_nodes)

        weighted_heuristic_methods(g_val, nodes_val, labels_val)  #  TODO?

    # get_nearest_neighbours()  # TODO: GNN methods don't work well

    # # doesn't depend on embeddings
    # dict_method_threshold = finetune_threshold_on_validation(g_val, nodes_val, labels_val, dict_method_threshold)
    # heuristic_methods(g_test, nodes_test, labels_test, g_val, dict_method_threshold)
    #
    # #### might erase later
    # g_train, g_test, examples_train, labels_train, examples_test, labels_test = test_train_split(g)
    # g_train, g_test, nodes_train, nodes_val, labels_train, labels_val, nodes_test, labels_test =\
    #     test_val_train_split(g)


if __name__ == '__main__':
    main()

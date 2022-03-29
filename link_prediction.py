from collections import Counter

from stellargraph.data import EdgeSplitter, BiasedRandomWalk, UnsupervisedSampler
from stellargraph.mapper import FullBatchLinkGenerator, Node2VecLinkGenerator, Node2VecNodeGenerator, \
    Attri2VecLinkGenerator, Attri2VecNodeGenerator, GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GCN, LinkEmbedding, Node2Vec, Attri2Vec, GraphSAGE
from stellargraph import StellarGraph
import tensorflow as tf
from tensorflow import keras
from stellargraph import datasets
from IPython.display import display, HTML
from sknetwork.linkpred import CommonNeighbors, JaccardIndex, SaltonIndex, HubPromotedIndex, AdamicAdar, \
    ResourceAllocation, PreferentialAttachment, HubDepressedIndex, whitened_sigmoid
from sknetwork.ranking import Katz
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
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
    # parameter settings for "p" and "q":
    p = 1.0
    q = 1.0
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


def finetune_threshold_on_validation(g_val, edge_ids_val, edge_labels_val):
    dict_method_threshold = {}
    for method_name in ["CommonNeighbors", "JaccardIndex", "SaltonIndex", "PreferentialAttachment", "AdamicAdar",
                        "HubPromotedIndex", "HubDepressedIndex", "ResourceAllocation"]:
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
        else:
            raise ValueError(f"method {method_name} nam not correct")

        adjacency = g_val.to_adjacency_matrix()
        method.fit_predict(adjacency, 0)  # assigns a scores to edges

        nodes_test = [list(g_val.node_ids_to_ilocs([edge_ids_val[i][0], edge_ids_val[i][1]])) for i in
                      range(len(edge_ids_val))]

        list_sym_predicted = []
        for (node1, node2), label in zip(nodes_test, edge_labels_val):
            common_neighbour_similarity = method.predict((node1, node2))
            list_sym_predicted.append(common_neighbour_similarity)

        max_accuracy = 0
        max_threshold = 0
        for threshold in np.linspace(0, 1, 20).tolist():
            predicted = whitened_sigmoid(np.asarray(list_sym_predicted)) > threshold  # TODO: keep whitened sigmoid?
            accuracy = accuracy_score(edge_labels_val, predicted)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_threshold = threshold
        dict_method_threshold[method_name] = max_threshold
        console.print(f"Method {method_name}, max accuracy: {max_accuracy} with threshold: {max_threshold}", style="magenta")
    return dict_method_threshold

def baselines(g_test, edge_ids_test, edge_labels_test, dict_method_threshold):
    print("Running baselines on test...")
    #TODO: Katz, Leich-Holme-Newman, Shortest Path?

    for method_name in ["CommonNeighbors", "JaccardIndex", "SaltonIndex", "PreferentialAttachment",  "AdamicAdar",
                        "HubPromotedIndex", "HubDepressedIndex", "ResourceAllocation"]:
        if not dict_method_threshold:
            threshold = 0.5
        else:
            threshold = dict_method_threshold[method_name] # finetuned on validation

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
        else:
            raise ValueError(f"method {method_name} nam not correct")

        console.print(f"Method {method_name}", style="magenta")
        adjacency = g_test.to_adjacency_matrix()
        method.fit_predict(adjacency, 0) # assigns a scores to edges
        # method.predict(list(range(adjacency.shape[0])))

        nodes_test = [list(g_test.node_ids_to_ilocs([edge_ids_test[i][0], edge_ids_test[i][1]])) for i in
                      range(len(edge_ids_test))]

        # predicted = []
        list_sym_predicted = []
        for (node1, node2), label in zip(nodes_test, edge_labels_test):
            #     print(node1, node2, label)
            common_neighbour_similarity = method.predict((node1, node2))
            list_sym_predicted.append(common_neighbour_similarity)
            # link = 0 if common_neighbour_similarity < threshold else 1
            # predicted.append(link)

        predicted = whitened_sigmoid(np.asarray(list_sym_predicted)) > threshold #TODO: keep whitened sigmoid?

        # print(f"Test GT data labels: {Counter(edge_labels_test)}")
        # print(f"Test Pred data sym scores: {Counter(list_sym_predicted)}")
        # print(f"Test Pred data labels: {Counter(predicted)}")

        accuracy = accuracy_score(edge_labels_test, predicted)
        console.print(f"Accuracy {accuracy:.2f} with method {method_name}", style="dim cyan")


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
    prediction = LinkEmbedding(activation="relu", method="dot")(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.binary_crossentropy,
        metrics=["binary_accuracy"],
    )
    # print(model.summary())

    # Train model
    history = model.fit(
        train_flow, validation_data=val_flow, callbacks=[callback], epochs=2, verbose=1, shuffle=False
    )

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

    return x_inp, x_out, train_flow, val_flow, test_flow, train_gen


def GraphSage_model(g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test):
    batch_size = 2
    layer_sizes = [20, 20]
    num_samples = [20, 10]

    train_gen = GraphSAGELinkGenerator(g_train, batch_size, num_samples)
    val_gen = GraphSAGELinkGenerator(g_val, batch_size, num_samples)
    test_gen = GraphSAGELinkGenerator(g_test, batch_size, num_samples)

    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.3
    )

    train_flow = train_gen.flow(nodes_train, labels_train)
    val_flow = val_gen.flow(nodes_val, labels_val)
    test_flow = test_gen.flow(nodes_test, labels_test)

    x_inp, x_out = graphsage.in_out_tensors()

    '''
        Model
    '''
    prediction = LinkEmbedding(activation="relu", method="dot")(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.binary_crossentropy,
        metrics=["binary_accuracy"],
    )
    # print(model.summary())

    # Train model
    history = model.fit(
        train_flow, validation_data=val_flow, callbacks=[callback], epochs=2, verbose=1, shuffle=False
    )

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

    return x_inp, x_out, train_flow, val_flow, test_flow, train_gen


def attri2vec_model(g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test):
    batch_size = 50
    layer_sizes = [128]

    train_gen = Attri2VecLinkGenerator(g_train, batch_size)
    val_gen = Attri2VecLinkGenerator(g_val, batch_size)
    test_gen = Attri2VecLinkGenerator(g_test, batch_size)

    attri2vec = Attri2Vec(
        layer_sizes=layer_sizes, generator=train_gen, bias=False, normalize=None
    )

    train_flow = train_gen.flow(nodes_train, labels_train)
    val_flow = val_gen.flow(nodes_val, labels_val)
    test_flow = test_gen.flow(nodes_test, labels_test)

    x_inp, x_out = attri2vec.in_out_tensors()
    '''
        Model
    '''
    prediction = LinkEmbedding(activation="relu", method="dot")(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.binary_crossentropy,
        metrics=["binary_accuracy"],
    )
    # print(model.summary())

    # Train model
    history = model.fit(
        train_flow, validation_data=val_flow, callbacks=[callback], epochs=2, verbose=1, shuffle=False
    )

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

    return x_inp, x_out, train_flow, val_flow, test_flow, train_gen


def node2vec_model(g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test):
    walk_number = 100
    walk_length = 5  # Larger values can be set to them to achieve better performance
    batch_size = 50
    emb_size = 128

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
    prediction = LinkEmbedding(activation="relu", method="dot")(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.binary_crossentropy,
        metrics=["binary_accuracy"],
    )
    # print(model.summary())

    # Train model
    history = model.fit(
        train_flow, validation_data=val_flow, callbacks=[callback], epochs=2, verbose=1, shuffle=False
    )

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

    return x_inp, x_out, train_flow, val_flow, test_flow, train_gen


def GNN_link_model(g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test):
    name = "GCN"
    print(f"Training {name}:")

    if name == "GCN":
        x_inp, x_out, train_flow, val_flow, test_flow, train_gen = GCN_model(g_train, g_val, g_test,
                                                                             nodes_train, nodes_val,
                                                                             nodes_test, labels_train,
                                                                             labels_val, labels_test)
    elif name == "graphsage":
        x_inp, x_out, train_flow, val_flow, test_flow, train_gen = GraphSage_model(g_train, g_val, g_test,
                                                                                   nodes_train, nodes_val,
                                                                                   nodes_test, labels_train,
                                                                                   labels_val, labels_test)
    elif name == "attri2vec":
        x_inp, x_out, train_flow, val_flow, test_flow, train_gen = attri2vec_model(g_train, g_val, g_test,
                                                                                   nodes_train, nodes_val,
                                                                                   nodes_test, labels_train,
                                                                                   labels_val, labels_test)
    elif name == "node2vec":
        x_inp, x_out, train_flow, val_flow, test_flow, train_gen = node2vec_model(g_train, g_val, g_test,
                                                                                  nodes_train, nodes_val,
                                                                                  nodes_test, labels_train,
                                                                                  labels_val, labels_test)
    else:
        raise ValueError("Wrong graph embeddings name!!")

    # fig = sg.utils.plot_history(history, return_figure=True)
    # # fig.savefig('data/plots/history_GCN_loss.pdf')
    #
    # train_metrics = model.evaluate(train_flow)
    # val_metrics = model.evaluate(val_flow)
    #
    # print(f"For operator {binary_op}: Train Set Metrics of the trained model:")
    # for name, val in zip(model.metrics_names, train_metrics):
    #     print("\t{}: {:0.4f}".format(name, val))
    #
    # print(f"For operator {binary_op}: Val Set Metrics of the trained model:")
    # for name, val in zip(model.metrics_names, val_metrics):
    #     print("\t{}: {:0.4f}".format(name, val))
    #
    # # if best_val < val:
    # #     best_op = binary_op
    # #     best_model = model
    # #
    # # test_metrics = best_model.evaluate(test_flow)
    # #
    # # print(f"For best operator, {best_op}: Test Set Metrics of the best trained model:")
    # # for name, val in zip(best_model.metrics_names, test_metrics):
    # #     print("\t{}: {:0.4f}".format(name, val))
    #
    # return prediction


def get_nearest_neighbours(g):
    square_node_data = pd.read_csv('data/graph/all_nodes.csv', index_col=0)
    square_node_data_values = square_node_data.values  # SentenceBert embeddings
    square_node_data_names = square_node_data.index.values.tolist()
    knn = NearestNeighbors(n_neighbors=6)
    knn.fit(square_node_data_values)

    # check_action = g.nodes()[1]
    check_action = "put tea bag"

    index_action = square_node_data_names.index(check_action)
    check_action_features = square_node_data_values[index_action].reshape(1, -1)
    print(f"SentenceBert Neighbours for: {check_action}:")
    list_indexes = knn.kneighbors(check_action_features, return_distance=False)
    print([square_node_data_names[index] for index in list_indexes[0]])

    # for gnn in ["GCN", "graphsage", "attri2vec", "node2vec"]:
    for gnn in ["GCN"]:
        with open('data/utils/' + gnn + '_node_embeddings.npy', 'rb') as f:
            gnn_node_embeddings = np.load(f)

        u_index = list(g.nodes()).index(check_action)
        check_action_features = gnn_node_embeddings[u_index].reshape(1, -1)
        knn = NearestNeighbors(n_neighbors=6)
        knn.fit(gnn_node_embeddings)
        print(f"{gnn} Neighbours for: {check_action}:")
        list_indexes = knn.kneighbors(check_action_features, return_distance=False)
        print([square_node_data_names[index] for index in list_indexes[0]])


def main():
    # g = test_cora()
    g = test_my_data(input_nodes='data/graph/all_avgclip_nodes.csv', input_edges='data/graph/all_edges.csv')

    g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train, labels_val, labels_test = \
        test_val_train_split2(g)

    # prediction = GNN_link_model(g_train, g_val, g_test, nodes_train, nodes_val, nodes_test, labels_train,
    #                                        labels_val, labels_test) # TODO: train models with more epochs and best params - embeddings depend on that too

    # dict_method_threshold = finetune_threshold_on_validation(g_val, nodes_val, labels_val)
    # dict_method_threshold = {}
    # baselines(g_test, nodes_test, labels_test, dict_method_threshold)

    # get_nearest_neighbours(g)     # TODO: compare with embeddings before and after model training

    ##### might erase later
    # g_train, g_test, examples_train, labels_train, examples_test, labels_test = test_train_split(g)
    # g_train, g_test, nodes_train, nodes_val, labels_train, labels_val, nodes_test, labels_test =\
    #     test_val_train_split(g)


if __name__ == '__main__':
    main()

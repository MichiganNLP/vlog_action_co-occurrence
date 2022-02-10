import stellargraph as sg
from stellargraph.data import EdgeSplitter, BiasedRandomWalk, UnsupervisedSampler
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.layer import GCN, LinkEmbedding
from stellargraph import StellarGraph
import tensorflow as tf
from tensorflow import keras
from stellargraph import datasets
from IPython.display import display, HTML
from sknetwork.linkpred import CommonNeighbors, JaccardIndex, SaltonIndex, HubPromotedIndex, AdamicAdar, ResourceAllocation
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

random_seed = 10
tf.keras.utils.set_random_seed(
    random_seed
)

def create_biased_random_walker(graph, walk_num, walk_length):
    # parameter settings for "p" and "q":
    p = 1.0
    q = 1.0
    return BiasedRandomWalk(graph, n=walk_num, length=walk_length, p=p, q=q)

def test_my_data():
    square_weight_edges = pd.read_csv('data/graph/all_edges.csv')
    square_node_data = pd.read_csv('data/graph/all_nodes.csv', index_col=0)
    G = StellarGraph(
        {"action": square_node_data}, {"line": square_weight_edges}
    )
    print(G.info())
    return G


def test_cora():
    dataset = datasets.Cora()
    display(HTML(dataset.description))
    G, _ = dataset.load(subject_as_feature=True)
    print(G.info())
    return G


def test_train_split(G):
    # Define an edge splitter on the original graph G:
    edge_splitter_test = EdgeSplitter(G)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
    # reduced graph G_test with the sampled links removed:
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global", keep_connected=True, seed=random_seed
    )

    # Define an edge splitter on the reduced graph G_test:
    edge_splitter_train = EdgeSplitter(G_test)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
    # reduced graph G_train with the sampled links removed:
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=0.1, method="global", keep_connected=True, seed=random_seed
    )

    return G_train, G_test, edge_ids_train, edge_labels_train, edge_ids_test, edge_labels_test


def baselines(G_test, edge_ids_test, edge_labels_test):
    print("Running baselines ...")
    # for method_name in ["CommonNeighbors", "JaccardIndex", "SaltonIndex", "HubPromotedIndex", "AdamicAdar"]:
    for method_name in ["CommonNeighbors", "JaccardIndex", "SaltonIndex", "HubPromotedIndex", "ResourceAllocation"]:
        threshold = 0.5
        if method_name == "CommonNeighbors":
            method = CommonNeighbors()
            threshold = 3  # TODO set it
        elif method_name == "JaccardIndex":
            method = JaccardIndex()
        elif method_name == "SaltonIndex":
            method = SaltonIndex()
        elif method_name == "HubPromotedIndex":
            method = HubPromotedIndex()
        elif method_name == "AdamicAdar":
            method = AdamicAdar()
        elif method_name == "ResourceAllocation":
            method = ResourceAllocation()
        else:
            raise ValueError(f"method {method_name} not correct")

        adjacency = G_test.to_adjacency_matrix()
        method.fit_predict(adjacency, 0)
        method.predict(list(range(adjacency.shape[0])))

        nodes_test = [list(G_test.node_ids_to_ilocs([edge_ids_test[i][0], edge_ids_test[i][1]])) for i in
                      range(len(edge_ids_test))]

        predicted = []
        for (node1, node2), label in zip(nodes_test, edge_labels_test):
            #     print(node1, node2, label)
            common_neighbour_similarity = method.predict((node1, node2))
            link = 0 if common_neighbour_similarity < threshold else 1
            predicted.append(link)

        accuracy = accuracy_score(edge_labels_test, predicted)
        print(f"Accuracy {accuracy} with method {method_name}")


def GCN_link_model(G_train, G_test, edge_ids_train, edge_labels_train, edge_ids_test, edge_labels_test):
    epochs = 50
    train_gen = FullBatchLinkGenerator(G_train, method="gcn")
    train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

    test_gen = FullBatchLinkGenerator(G_test, method="gcn")
    test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

    gcn = GCN(
        layer_sizes=[16, 16], activations=["relu", "relu"], generator=train_gen, dropout=0.3
    )
    x_inp, x_out = gcn.in_out_tensors()

    prediction = LinkEmbedding(activation="relu", method="ip")(x_out)

    model = keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.binary_crossentropy,
        metrics=["binary_accuracy"],
    )
    print(model.summary())

    # Evaluate initial, untrained model on train and test sets
    init_train_metrics = model.evaluate(train_flow)
    init_test_metrics = model.evaluate(test_flow)
    print("\nTrain Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    # Train model
    history = model.fit(
        train_flow, epochs=epochs, validation_data=test_flow, verbose=0, shuffle=False
    )
    fig = sg.utils.plot_history(history, return_figure=True)
    fig.savefig('data/plots/history_GCN_loss.pdf')

    train_metrics = model.evaluate(train_flow)
    test_metrics = model.evaluate(test_flow)

    print("\nTrain Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    return prediction

def gcn_embedding(graph, name):

    # Set the embedding dimensions and walk number:
    dimensions = [128, 128]
    walk_number, walk_length, epochs, batch_size = 1, 5, 6, 50
    # walk_number, walk_length, epochs, batch_size = 1, 5, 1, 50 #for test reduce nb epochs

    print(f"Training GCN for '{name}':")

    graph_node_list = list(graph.nodes())

    # Create the biased random walker to generate random walks
    walker = create_biased_random_walker(graph, walk_number, walk_length)

    # Create the unsupervised sampler to sample (target, context) pairs from random walks
    unsupervised_samples = UnsupervisedSampler(
        graph, nodes=graph_node_list, walker=walker
    )

    # Define a GCN training generator, which generates the full batch of training pairs
    generator = FullBatchLinkGenerator(graph, method="gcn")

    # Create the GCN model
    gcn = GCN(
        layer_sizes=dimensions,
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.3,
    )

    # Build the model and expose input and output sockets of GCN, for node pair inputs
    x_inp, x_out = gcn.in_out_tensors()

    # Use the dot product of node embeddings to make node pairs co-occurring in short random walks represented closely
    prediction = LinkEmbedding(activation="sigmoid", method="ip")(x_out)
    prediction = keras.layers.Reshape((-1,))(prediction)

    # Stack the GCN encoder and prediction layer into a Keras model, and specify the loss
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    # Train the model
    batches = unsupervised_samples.run(batch_size)
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        batch_iter = 1
        for batch in batches:
            samples = generator.flow(batch[0], targets=batch[1], use_ilocs=True)[0]
            [loss, accuracy] = model.train_on_batch(x=samples[0], y=samples[1])
            output = (
                f"{batch_iter}/{len(batches)} - loss:"
                + " {:6.4f}".format(loss)
                + " - binary_accuracy:"
                + " {:6.4f}".format(accuracy)
            )
            if batch_iter == len(batches):
                print(output)
            else:
                print(output, end="\r")
            batch_iter = batch_iter + 1

    # Get representations for all nodes in ``graph``
    embedding_model = keras.Model(inputs=x_inp, outputs=x_out)
    node_embeddings = embedding_model.predict(
        generator.flow(list(zip(graph_node_list, graph_node_list)))
    )
    node_embeddings = node_embeddings[0][:, 0, :]
    np.save("data/utils/GCN_node_embeddings.npy", node_embeddings)

    def get_embedding(u):
        u_index = graph_node_list.index(u)
        return node_embeddings[u_index]

    return get_embedding


def get_nearest_neighbours(G):
    square_node_data = pd.read_csv('data/graph/all_nodes.csv', index_col=0)
    square_node_data_values = square_node_data.values   #SentenceBert embeddings
    square_node_data_names = square_node_data.index.values.tolist()
    knn = NearestNeighbors(n_neighbors=6)
    knn.fit(square_node_data_values)


    # check_action = square_node_data_names[0]
    # check_action_features = square_node_data_values[0].reshape(1, -1)

    check_action = "drink rest of tea"
    index_action = square_node_data_names.index(check_action)
    check_action_features = square_node_data_values[index_action].reshape(1, -1)
    print(f"SentenceBert Neighbours for {check_action}:")
    list_indexes = knn.kneighbors(check_action_features, return_distance=False)
    print([square_node_data_names[index] for index in list_indexes[0]])

    #TODO GCN embeddings
    with open('data/utils/GCN_node_embeddings.npy', 'rb') as f:
        GCN_node_embeddings = np.load(f)
    graph_node_list = list(G.nodes())
    u_index = graph_node_list.index(check_action)
    check_action_features = GCN_node_embeddings[u_index].reshape(1, -1)
    knn = NearestNeighbors(n_neighbors=6)
    knn.fit(GCN_node_embeddings)
    print(f"GCN Neighbours for {check_action}:")
    list_indexes = knn.kneighbors(check_action_features, return_distance=False)
    print([square_node_data_names[index] for index in list_indexes[0]])


def main():
    # G = test_cora()
    G = test_my_data()
    # G_train, G_test, edge_ids_train, edge_labels_train, edge_ids_test, edge_labels_test = test_train_split(G)
    # GCN_link_model(G_train, G_test, edge_ids_train, edge_labels_train, edge_ids_test, edge_labels_test)
    # baselines(G_test, edge_ids_test, edge_labels_test)

    gcn_embedding(G, name="my data")
    get_nearest_neighbours(G)    #TODO try SentenceBert vs. GCN embeddings

if __name__ == '__main__':
    main()

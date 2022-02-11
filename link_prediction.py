import stellargraph as sg
from stellargraph.data import EdgeSplitter, BiasedRandomWalk, UnsupervisedSampler
from stellargraph.mapper import FullBatchLinkGenerator, Node2VecLinkGenerator, Node2VecNodeGenerator, \
    Attri2VecLinkGenerator, Attri2VecNodeGenerator, GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GCN, LinkEmbedding, Node2Vec, link_classification, Attri2Vec, GraphSAGE
from stellargraph import StellarGraph
import tensorflow as tf
from tensorflow import keras
from stellargraph import datasets
from IPython.display import display, HTML
from sknetwork.linkpred import CommonNeighbors, JaccardIndex, SaltonIndex, HubPromotedIndex, AdamicAdar, \
    ResourceAllocation
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
import numpy as np

random_seed = 10
tf.keras.utils.set_random_seed(
    random_seed
)

''' Hyperparameters '''
walk_length, epochs, batch_size = 5, 6, 50
# walk_length, epochs, batch_size = 5, 1, 50


def create_biased_random_walker(graph, walk_num, walk_length):
    # parameter settings for "p" and "q":
    p = 1.0
    q = 1.0
    return BiasedRandomWalk(graph, n=walk_num, length=walk_length, p=p, q=q)


def test_my_data():
    square_weight_edges = pd.read_csv('data/graph/all_edges.csv')
    square_node_data = pd.read_csv('data/graph/all_nodes.csv', index_col=0)
    G = StellarGraph(
        {"action": square_node_data}, {"co-occurs": square_weight_edges}
    )
    print(G.info())
    return G


def test_cora():
    dataset = datasets.Cora()
    display(HTML(dataset.description))
    G, _ = dataset.load(subject_as_feature=True)
    print(G.info())
    return G


def test_val_train_split(G):
    # Define an edge splitter on the original graph G:
    edge_splitter_test = EdgeSplitter(G)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
    # reduced graph G_test with the sampled links removed:
    G_test, examples_test, labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global", keep_connected=True, seed=random_seed
    )

    # Define an edge splitter on the reduced graph G_test:
    edge_splitter_train = EdgeSplitter(G_test)

    G_train, examples, labels = edge_splitter_train.train_test_split(
        p=0.1, method="global"
    )
    (
        examples_train,
        examples_model_selection,  # validation
        labels_train,
        labels_model_selection,
    ) = train_test_split(examples, labels, train_size=0.75, test_size=0.25)

    print(G_train.info())
    print(len(examples_test), len(examples_train), len(examples_model_selection))
    return G_train, G_test, examples_train, examples_model_selection, labels_train, labels_model_selection, examples_test, labels_test


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
    print("Running baselines on test...")
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
        print(f"Accuracy {accuracy:.2f} with method {method_name}")


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


def node2vec_embedding(graph, name):
    # Set the embedding dimension and walk number:
    dimension = 128
    walk_number = 20

    print(f"Training Node2Vec for '{name}':")

    graph_node_list = list(graph.nodes())

    # Create the biased random walker to generate random walks
    walker = create_biased_random_walker(graph, walk_number, walk_length)

    # Create the unsupervised sampler to sample (target, context) pairs from random walks
    unsupervised_samples = UnsupervisedSampler(
        graph, nodes=graph_node_list, walker=walker
    )

    # Define a Node2Vec training generator, which generates batches of training pairs
    generator = Node2VecLinkGenerator(graph, batch_size)

    # Create the Node2Vec model
    node2vec = Node2Vec(dimension, generator=generator)

    # Build the model and expose input and output sockets of Node2Vec, for node pair inputs
    x_inp, x_out = node2vec.in_out_tensors()

    # Use the link_classification function to generate the output of the Node2Vec model
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="dot"
    )(x_out)

    # Stack the Node2Vec encoder and prediction layer into a Keras model, and specify the loss
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    # Train the model
    model.fit(
        generator.flow(unsupervised_samples),
        epochs=epochs,
        verbose=2,
        use_multiprocessing=False,
        workers=4,
        shuffle=True,
    )

    # Build the model to predict node representations from node ids with the learned Node2Vec model parameters
    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    # Get representations for all nodes in ``graph``
    node_gen = Node2VecNodeGenerator(graph, batch_size).flow(graph_node_list)
    node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=0)

    def get_embedding(u):
        u_index = graph_node_list.index(u)
        return node_embeddings[u_index]

    return get_embedding


def attri2vec_embedding(graph, name):
    # Set the embedding dimension and walk number:
    dimension = [128]
    walk_number = 4

    print(f"Training Attri2Vec for '{name}':")

    graph_node_list = list(graph.nodes())

    # Create the biased random walker to generate random walks
    walker = create_biased_random_walker(graph, walk_number, walk_length)

    # Create the unsupervised sampler to sample (target, context) pairs from random walks
    unsupervised_samples = UnsupervisedSampler(
        graph, nodes=graph_node_list, walker=walker
    )

    # Define an Attri2Vec training generator, which generates batches of training pairs
    generator = Attri2VecLinkGenerator(graph, batch_size)

    # Create the Attri2Vec model
    attri2vec = Attri2Vec(
        layer_sizes=dimension, generator=generator, bias=False, normalize=None
    )

    # Build the model and expose input and output sockets of Attri2Vec, for node pair inputs
    x_inp, x_out = attri2vec.in_out_tensors()

    # Use the link_classification function to generate the output of the Attri2Vec model
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)

    # Stack the Attri2Vec encoder and prediction layer into a Keras model, and specify the loss
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    # Train the model
    model.fit(
        generator.flow(unsupervised_samples),
        epochs=epochs,
        verbose=2,
        use_multiprocessing=False,
        workers=1,
        shuffle=True,
    )

    # Build the model to predict node representations from node features with the learned Attri2Vec model parameters
    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    # Get representations for all nodes in ``graph``
    node_gen = Attri2VecNodeGenerator(graph, batch_size).flow(graph_node_list)
    node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=0)

    def get_embedding(u):
        u_index = graph_node_list.index(u)
        return node_embeddings[u_index]

    return get_embedding


def graphsage_embedding(graph, name):
    # Set the embedding dimensions, the numbers of sampled neighboring nodes and walk number:
    dimensions = [128, 128]
    num_samples = [10, 5]
    walk_number = 1

    print(f"Training GraphSAGE for '{name}':")

    graph_node_list = list(graph.nodes())

    # Create the biased random walker to generate random walks
    walker = create_biased_random_walker(graph, walk_number, walk_length)

    # Create the unsupervised sampler to sample (target, context) pairs from random walks
    unsupervised_samples = UnsupervisedSampler(
        graph, nodes=graph_node_list, walker=walker
    )

    # Define a GraphSAGE training generator, which generates batches of training pairs
    generator = GraphSAGELinkGenerator(graph, batch_size, num_samples)

    # Create the GraphSAGE model
    graphsage = GraphSAGE(
        layer_sizes=dimensions,
        generator=generator,
        bias=True,
        dropout=0.0,
        normalize="l2",
    )

    # Build the model and expose input and output sockets of GraphSAGE, for node pair inputs
    x_inp, x_out = graphsage.in_out_tensors()

    # Use the link_classification function to generate the output of the GraphSAGE model
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)

    # Stack the GraphSAGE encoder and prediction layer into a Keras model, and specify the loss
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    # Train the model
    model.fit(
        generator.flow(unsupervised_samples),
        epochs=epochs,
        verbose=2,
        use_multiprocessing=False,
        workers=4,
        shuffle=True,
    )

    # Build the model to predict node representations from node features with the learned GraphSAGE model parameters
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    # Get representations for all nodes in ``graph``
    node_gen = GraphSAGENodeGenerator(graph, batch_size, num_samples).flow(
        graph_node_list
    )
    node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=0)

    def get_embedding(u):
        u_index = graph_node_list.index(u)
        return node_embeddings[u_index]

    return get_embedding


def gcn_embedding(graph, name):
    # Set the embedding dimensions and walk number:
    dimensions = [128, 128]
    walk_number = 1
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
        print(f"Epoch: {epoch + 1}/{epochs}")
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
    square_node_data_values = square_node_data.values  # SentenceBert embeddings
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

    # TODO GCN embeddings
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


'''
1. link embeddings
We calculate link/edge embeddings for the positive and negative edge samples by applying a binary operator 
on the embeddings of the source and target nodes of each sampled edge
'''


def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [
        binary_operator(transform_node(src), transform_node(dst))
        for src, dst in link_examples
    ]


'''
2. training classifier
Given the embeddings of the positive and negative examples, we train a logistic regression classifier
to predict a binary value indicating whether an edge between two nodes should exist or not. 
'''


def train_link_prediction_model(
        link_examples, link_labels, get_embedding, binary_operator
):
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(
        link_examples, get_embedding, binary_operator
    )
    clf.fit(link_features, link_labels)
    return clf


def link_prediction_classifier(max_iter=5000):
    # learning_rate_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    learning_rate_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="accuracy", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", learning_rate_clf)])


'''
3. evaluate classifier
We evaluate the performance of the link classifier for each of the 4 operators on the training data 
with node embeddings calculated on the Train Graph, and select the best classifier. 
The best classifier is then used to calculate scores on the test data with node embeddings trained on the Train Graph
'''


def evaluate_link_prediction_model(
        clf, link_examples_test, link_labels_test, get_embedding, binary_operator
):
    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator
    )
    # score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    score = evaluate_accuracy(clf, link_features_test, link_labels_test)
    return score

def evaluate_accuracy(clf, link_features, link_labels):
    # predicted = clf.predict_proba(link_features)
    #
    # # check which class corresponds to positive links
    # positive_column = list(clf.classes_).index(1)
    # predicted_threshold = [0 if score < 0.5 else 1 for score in predicted[:, positive_column]]
    # print(predicted_threshold)
    # return accuracy_score(link_labels, predicted_threshold)
    predicted = clf.predict(link_features)
    return accuracy_score(link_labels, predicted)

def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)
    positive_column = list(clf.classes_).index(1)

    # check which class corresponds to positive links
    return roc_auc_score(link_labels, predicted[:, positive_column])


def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def operator_avg(u, v):
    return (u + v) / 2.0


def run_link_prediction(binary_operator, embedding_train, examples_train, labels_train, examples_model_selection,
                        labels_model_selection):
    clf = train_link_prediction_model(
        examples_train, labels_train, embedding_train, binary_operator
    )
    score = evaluate_link_prediction_model(
        clf,
        examples_model_selection,
        labels_model_selection,
        embedding_train,
        binary_operator,
    )

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score": score,
    }


def train_and_evaluate(embedding, name, G_train, examples_train, labels_train, examples_model_selection,
                       labels_model_selection, examples_test, labels_test, binary_operators):
    embedding_train = embedding(G_train, "Train Graph")

    # Train the link classification model with the learned embedding
    results = [run_link_prediction(op, embedding_train, examples_train, labels_train, examples_model_selection,
                                   labels_model_selection) for op in binary_operators]
    best_result = max(results, key=lambda result: result["score"])
    print(
        f"\nBest result with '{name}' embeddings from '{best_result['binary_operator'].__name__}'"
    )
    display(
        pd.DataFrame(
            [(result["binary_operator"].__name__, result["score"]) for result in results],
            # columns=("name", "ROC AUC"),
            columns=("name", "Accuracy"),
        ).set_index("name")
    )

    # Evaluate the best model using the test set
    test_score = evaluate_link_prediction_model(
        best_result["classifier"],
        examples_test,
        labels_test,
        embedding_train,
        best_result["binary_operator"],
    )

    return test_score


def main():
    # G = test_cora()
    G = test_my_data()
    # G_train, G_test, examples_train, labels_train, examples_test, labels_test = test_train_split(G)
    G_train, G_test, examples_train, examples_model_selection, labels_train, labels_model_selection, \
    examples_test, labels_test = test_val_train_split(G)

    binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]

    node2vec_result = train_and_evaluate(node2vec_embedding, "Node2Vec", G_train, examples_train, labels_train,
                                         examples_model_selection, labels_model_selection, examples_test,
                                         labels_test, binary_operators)
    attri2vec_result = train_and_evaluate(attri2vec_embedding, "Attri2Vec", G_train, examples_train, labels_train,
                                         examples_model_selection, labels_model_selection, examples_test,
                                         labels_test, binary_operators)
    graphsage_result = train_and_evaluate(graphsage_embedding, "GraphSAGE", G_train, examples_train, labels_train,
                                         examples_model_selection, labels_model_selection, examples_test,
                                         labels_test, binary_operators)
    gcn_result = train_and_evaluate(gcn_embedding, "GCN", G_train, examples_train, labels_train,
                                         examples_model_selection, labels_model_selection, examples_test,
                                         labels_test, binary_operators)

    df = pd.DataFrame(
        [
            ("Node2Vec", node2vec_result),
            ("Attri2Vec", attri2vec_result),
            ("GraphSAGE", graphsage_result),
            ("GCN", gcn_result),
        ],
        # columns=("name", "ROC AUC"),
        columns=("name", "Accuracy"),
    ).set_index("name")
    display(df)

    # GCN_link_model(G_train, G_test, examples_train, labels_train, examples_test, labels_test)
    # baselines(G_test, examples_test, labels_test)

    # gcn_embedding(G, name="my data")
    # get_nearest_neighbours(G)    #TODO try SentenceBert vs. GCN embeddings


if __name__ == '__main__':
    main()

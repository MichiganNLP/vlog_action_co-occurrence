import pandas as pd
import logging
import torch
import sklearn
import json
import ast
import random
from collections import Counter

cuda_available = torch.cuda.is_available()

from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs
)
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def prep_data_for_sentence_classif():
    with open('data/final_data/dict_format_data.json') as json_file:
        dict_format_data = json.load(json_file)

    list_all_data_actions = []
    all_together_data = dict_format_data["together"]
    all_not_together_data = dict_format_data["not_together"]

    count_0 = 0
    count_1 = 0
    for dict in [all_together_data, all_not_together_data]:
        if dict == all_not_together_data:
            label = 0
        else:
            label = 1
        for action_tuple in dict.keys():
            # list_video_sentence_tuple = dict[action_tuple]
            action_tuple = ast.literal_eval(action_tuple)
            list_all_data_actions.append([action_tuple[0], action_tuple[1], label])
            # for video_sentence_tuple in list_video_sentence_tuple:
            #     (video, sentence) = ast.literal_eval(video_sentence_tuple)
            #     list_all_data_actions.append([sentence[0], sentence[1], label])
            if label:
                count_1 += 1
            else:
                count_0 += 1

    print(count_0, count_1)

    # make action tuples together and not together equal number
    list_all_data_actions = list_all_data_actions[: 2971 + 2971]
    labels = [l[2] for l in list_all_data_actions]
    count_0 = Counter(labels)[0]
    count_1 = Counter(labels)[1]
    print(Counter(labels), count_0, count_1)

    total_data_nb = count_0 + count_1
    total_train_nb = int(total_data_nb * 0.9)
    train_data, eval_data = [], []
    count_0_train, count_0_eval, count_1_train, count_1_eval = 0, 0, 0, 0
    for l in list_all_data_actions:
        if l[2] == 0 and count_0_train < total_train_nb // 2:
            train_data.append(l)
            count_0_train += 1
        else:
            if l[2] == 0:
                eval_data.append(l)
                count_0_eval += 1
        if l[2] == 1 and count_1_train < total_train_nb // 2:
            train_data.append(l)
            count_1_train += 1
        else:
            if l[2] == 1:
                eval_data.append(l)
                count_1_eval += 1

    print("train: ", count_0_train, count_1_train)
    # print(train_data)
    print("eval: ", count_0_eval, count_1_eval)
    # print(eval_data)
    assert(count_0 == count_1)
    random.shuffle(train_data)
    random.shuffle(eval_data)
    return train_data, eval_data

def sentence_pair_classif():
    # Preparing train & eval data
    train_data, eval_data = prep_data_for_sentence_classif()

    train_df = pd.DataFrame(train_data)
    train_df.columns = ["text_a", "text_b", "labels"]

    eval_df = pd.DataFrame(eval_data)
    eval_df.columns = ["text_a", "text_b", "labels"]

    # Optional model configuration
    model_args = ClassificationArgs(num_train_epochs=10, early_stopping_patience=5, overwrite_output_dir=True)
    # Create a ClassificationModel
    model = ClassificationModel("roberta", "roberta-base", args=model_args, use_cuda=cuda_available)
    # Train the model
    model.train_model(train_df, acc=sklearn.metrics.accuracy_score)
    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(
        eval_df, acc=sklearn.metrics.accuracy_score
    )
    print(result)

    # # Make predictions with the model
    # predictions, raw_outputs = model.predict(
    #     [
    #         [
    #             "Legolas was an expert archer",
    #             "Legolas was taller than Gimli",
    #         ]
    #     ]
    # )
    # print(predictions, raw_outputs, result)

def main():
    # prep_data_for_sentence_classif()
    #TODO: only actions, only sentences, action prompt - action1 happens before action2 ?
    sentence_pair_classif()


if __name__ == '__main__':
    main()
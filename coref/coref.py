import json
from tqdm import tqdm
import spacy
import neuralcoref

# FILE_NAME = "dict_example2_actions.json"
FILE_NAME = "all_sentence_transcripts.json"
# PATH_IN = "/home/oana/Oana/Action_Recog/vlog_action_order/data/analyse_verbs/"
PATH_IN = "/home/oana/Oana/Action_Recog/vlog_action_order/data/"

print("Spacy version: ", spacy.__version__)
nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

def main():
    with open(PATH_IN + FILE_NAME) as json_file:
        data = json.load(json_file)
    resolved = 0
    count = 0 
    # for key in list(data.keys()):
    #     labels = data[key]
    #     for label in labels:
    #         doc = nlp(label["sentence"])
    #         label["resolved_sentence"] = doc._.coref_resolved
    #         count += 1
    #         if label["resolved_sentence"] != label["sentence"]:
    #             print(label["resolved_sentence"])
    #             print(label["sentence"])
    #             resolved += 1

    coref_data = {}
    for key in tqdm(list(data.keys())):
        labels = data[key]
        coref_data[key] = []
        for i, [sentence, time_s, time_e] in enumerate(labels):
            doc = nlp(sentence)
            resolved_sentence = doc._.coref_resolved
            count += 1
            list_mentions = []
            if sentence != resolved_sentence:
                # print(sentence)
                # print(resolved_sentence)
                resolved += 1
            # data[key][i].append(resolved_sentence)
                for cluster in doc._.coref_clusters:
                    list_mentions.append(cluster.mentions)
                # print(list_mentions)
            coref_data[key].append({"sentence": sentence, "coref_sentence": resolved_sentence, "list_mentions":str(list_mentions), "time_s": time_s, "time_e":time_e})
    # print("resolved out of total: ", resolved, count)
    with open(PATH_IN + "coref_" + FILE_NAME, 'w') as outfile:
        json.dump(coref_data, outfile)


if __name__ == '__main__':
    main()
    
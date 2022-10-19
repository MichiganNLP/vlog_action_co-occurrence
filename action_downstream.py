import json





def read_tomm_actions():
    with open("data/action_localization/dict_all_annotations_1_10channels.json") as f:
        dict_all_annotations_1_10channels = json.loads(f.read())




def get_tomm_action_embeddings():



if __name__ == '__main__':
    main()
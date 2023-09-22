Human Action Co-occurrence in Lifestyle Vlogs using Graph Link Prediction
=================================================================================
Oana Ignat, Santiago Castro, Weiji Li and Rada Mihalcea

[Language and Information Technologies (LIT), University of Michigan](https://lit.eecs.umich.edu/)

This repository contains the dataset and code for our paper:

[Human Action Co-occurrence in Lifestyle Vlogs with Graph Link Prediction](https://arxiv.org/abs/2309.06219)

## Task Description

![img/graph_intro.png](img/graph_intro.png)

Human action co-occurrence in lifestyle vlogs: two actions co-occur if they occur in the same
interval of time (10 seconds) in a video. The actions are represented as nodes in a graph, 
the co-occurrence relation between two actions is represented through a link between the actions, 
and the action co-occurrence identification task as a link prediction task.

## Data

- The resources for creating the data are in [`data/utils`](data/utils). More details in [`data_processing.py`](data_processing.py)
- The graph is saved in [`data/graph/edges.csv`](data/graph/edges.csv)
- The node embeddings are saved in [`data/graph/{default_feat_nodes}_nodes.csv`](data/graph/{default_feat_nodes}_nodes.csv)  
where default_feat_nodes can be: "txt_action", "txt_transcript", "vis_action", "vis_video", "vis_action_video". 
More details in [`link_prediction.py`](link_prediction.py).

-  Sample frames 4 frames per video and their action label are found in [`frames_sample`](frames_sample)


## Features
- The visual features are in [`data/clip_features.pt`](data/clip_features.pt)
- The textual and graph features are in [`data/graph`](data/graph) and are computed in [`data_processing.py`](data_processing.py), function save_nodes_edges_df

## Setup
```bash
conda env create
conda activate action_order
pip install -r requirements.txt

spacy download en_core_web_sm
spacy download en_core_web_trf
```

## Experiments
+ Run data collection and processing from [`data_processing.py`](data_processing.py) 
+ Run action co-occurrence/ link prediction models from [`link_prediction.py`](link_prediction.py)
+ Run downstream task experiments from [`action_downstream.py`](action_downstream.py) and from 
get_nearest_neighbours function in [`link_prediction.py`](link_prediction.py)
+ Run data analyses from [`data_analysis.ipynb`](data_analysis.ipynb)
+ Run video related scripts from [`utils`](utils)


## Citation information
If you use this dataset or any ideas based on the associated research article, please cite the following:

```bibtex
@misc{ignat2023human,
      title={Human Action Co-occurrence in Lifestyle Vlogs using Graph Link Prediction}, 
      author={Oana Ignat and Santiago Castro and Weiji Li and Rada Mihalcea},
      year={2023},
      eprint={2309.06219},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

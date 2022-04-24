#!/usr/bin/env python
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEMPERATURE = 100


def main() -> None:
    features_dict = torch.load("data/clip_features3.pt")

    with torch.inference_mode():
        video_features = torch.stack([b["visual"] for b in features_dict.values()]).to(DEVICE)
        text_features = torch.stack([b["text"] for b in features_dict.values()]).to(DEVICE)

        video_features /= video_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = video_features @ text_features.T

        target = torch.arange(len(similarity), device="cuda")

        sorted_predicted_positions = similarity.argsort(dim=-1, descending=True)
        ranks = (sorted_predicted_positions == target.unsqueeze(dim=-1)).nonzero(as_tuple=True)[1]  # noqa

        print("Mean rank:", ranks.to(torch.float).mean())
        print("Median rank:", ranks.median())

        print("R@1:", (similarity.argmax(dim=-1) == target).to(torch.float).mean())
        print("R@5:", (similarity.topk(5)[1] == target.unsqueeze(dim=-1)).sum(dim=-1).to(torch.float).mean())  # noqa
        print("R@10:", (similarity.topk(10)[1] == target.unsqueeze(dim=-1)).sum(dim=-1).to(torch.float).mean())  # noqa

        probs = (similarity * TEMPERATURE).softmax(dim=-1)

        ground_truth_probs = probs.diag()

        print("Mean ground truth probability:", ground_truth_probs.mean())
        print("Median ground truth probability:", ground_truth_probs.median())


if __name__ == '__main__':
    main()

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

        probs = (similarity * TEMPERATURE).softmax(dim=-1)

        target = torch.arange(len(probs), device="cuda")

        sorted_predicted_positions = probs.argsort(dim=-1, descending=True)
        ranks = (sorted_predicted_positions == target.unsqueeze(dim=-1)).nonzero(as_tuple=True)[1]  # noqa

        print("Mean rank:", ranks.to(torch.float).mean())
        print("Median rank:", ranks.median())

        print("R@1:", (probs.argmax(dim=-1) == target).to(torch.float).mean())
        print("R@5:", (probs.topk(5)[1] == target.unsqueeze(dim=-1)).sum(dim=-1).to(torch.float).mean())  # noqa
        print("R@10:", (probs.topk(10)[1] == target.unsqueeze(dim=-1)).sum(dim=-1).to(torch.float).mean())  # noqa


if __name__ == '__main__':
    main()

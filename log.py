import json
import torch
import os


class Logger:
    def __init__(self, name):
        root = os.path.join("log", name)
        os.makedirs(root, exist_ok=True)

        self.metrics_file = open(os.path.join(root, "metrics.jsonl"), "w")
        self.model_file = os.path.join(root, "model.pt")

    def log(self, model=None, **metrics):
        serializable = {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
        self.metrics_file.write(json.dumps(serializable) + "\n")
        self.metrics_file.flush()

        if model is not None:
            torch.save(model.state_dict(), self.model_file)
import json
import torch
import os


class Logger:
    def __init__(self, name):
        self.root = os.path.join("log", name)
        os.makedirs(self.root, exist_ok=True)

        self.metrics_file = open(os.path.join(self.root, "metrics.jsonl"), "w")
        self.model_file = os.path.join(self.root, "model.pt")
        self._extra_files = {}

    def log(self, model=None, **metrics):
        serializable = {
            k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()
        }
        self.metrics_file.write(json.dumps(serializable) + "\n")
        self.metrics_file.flush()

        if model is not None:
            torch.save(model.state_dict(), self.model_file)

    def log_to(self, filename, **record):
        """Append a record to a side-channel JSONL (e.g. per-generation ELA
        features), kept separate from the main metrics stream."""
        f = self._extra_files.get(filename)
        if f is None:
            f = self._extra_files[filename] = open(os.path.join(self.root, filename), "w")
        serializable = {
            k: v.item() if hasattr(v, "item") else v for k, v in record.items()
        }
        f.write(json.dumps(serializable) + "\n")
        f.flush()

import numpy as np
import torch
import torch.nn as nn
import tqdm
from algorithms._common import get_params, set_params, weight_stats


class _LowRankWrapper(nn.Module):
    def __init__(self, model, rank):
        super().__init__()
        self.model = model
        self.ndim = sum(p.numel() for p in model.parameters())
        self.n = int(np.ceil(np.sqrt(self.ndim)))
        self.rank = rank

        params = get_params(model)
        padded = np.zeros(self.n * self.n)
        padded[: len(params)] = params
        U_np, s_np, Vt_np = np.linalg.svd(
            padded.reshape(self.n, self.n), full_matrices=False
        )
        sq_s = np.sqrt(s_np[:rank])
        self.U = nn.Parameter(torch.tensor(U_np[:, :rank] * sq_s, dtype=torch.float32))
        self.V = nn.Parameter(torch.tensor(Vt_np[:rank].T * sq_s, dtype=torch.float32))

        for p in self.model.parameters():
            p.requires_grad_(False)

    def _flat_params(self):
        return (self.U @ self.V.T).ravel()[: self.ndim]

    def forward(self, x):
        flat = self._flat_params()
        param_dict, offset = {}, 0
        for name, p in self.model.named_parameters():
            numel = p.numel()
            param_dict[name] = flat[offset : offset + numel].reshape(p.shape)
            offset += numel
        return torch.func.functional_call(self.model, param_dict, x)

    def write_back(self):
        set_params(self.model, self._flat_params().detach().numpy())


def run(
    model,
    criterion,
    X_train,
    y_train,
    X_val,
    y_val,
    num_epochs,
    weight_decay,
    logger,
    lr=3e-2,
    rank=0,
):
    if rank > 0:
        net = _LowRankWrapper(model, rank)
    else:
        net = model

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    pbar = tqdm.trange(num_epochs, leave=True, position=0)
    for epoch in pbar:
        optimizer.zero_grad()
        y_pred = net(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_acc = (y_pred.argmax(dim=1) == y_train).float().mean() * 100
            y_pred_val = net(X_val)
            val_loss = criterion(y_pred_val, y_val)
            val_acc = (y_pred_val.argmax(dim=1) == y_val).float().mean() * 100

        pbar.set_description(
            f"{loss:10.2f}, {train_acc:>3.0f} | {val_loss:>8.2f}, {val_acc:>4.0f}"
        )
        if logger is not None:
            if rank > 0:
                net.write_back()
            logger.log(
                model=model,
                epoch=epoch,
                train_loss=loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                **weight_stats(model),
            )

    if rank > 0:
        net.write_back()

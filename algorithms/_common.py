import torch
import numpy as np
import tqdm


def get_params(model):
    return np.concatenate([p.detach().numpy().ravel() for p in model.parameters()])


def set_params(model, x):
    x_t = torch.from_numpy(x.astype(np.float32))
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(x_t[idx : idx + n].reshape(p.shape))
            idx += n


def make_fitness(
    model,
    criterion,
    X_train,
    y_train,
    X_val,
    y_val,
    weight_decay,
    logger,
    pop_size,
    num_epochs,
):
    eval_count = [0]
    best_loss = [float("inf")]
    best_x = [get_params(model)]
    pbar = tqdm.tqdm(total=num_epochs, leave=True, position=0)

    def fitness(x):
        set_params(model, x)
        with torch.no_grad():
            logits = model(X_train)
            loss = criterion(logits, y_train).item() + np.mean(x**2) * weight_decay

        if loss < best_loss[0]:
            best_loss[0] = loss
            best_x[0] = x.copy()

        eval_count[0] += 1
        if eval_count[0] % pop_size == 0:
            gen = eval_count[0] // pop_size
            set_params(model, best_x[0])
            with torch.no_grad():
                tr_logits = model(X_train)
                tr_loss = criterion(tr_logits, y_train)
                tr_acc = (tr_logits.argmax(1) == y_train).float().mean() * 100
                vl_logits = model(X_val)
                vl_loss = criterion(vl_logits, y_val)
                vl_acc = (vl_logits.argmax(1) == y_val).float().mean() * 100
            pbar.set_description(
                f"{tr_loss:10.2f}, {tr_acc:>3.0f} | {vl_loss:>8.2f}, {vl_acc:>4.0f}"
            )
            pbar.update(1)
            if logger is not None:
                logger.log(
                    model=model,
                    epoch=gen,
                    train_loss=tr_loss,
                    train_acc=tr_acc,
                    val_loss=vl_loss,
                    val_acc=vl_acc,
                )

        return loss

    return fitness, pbar, best_x

import math
import random
import warnings
from functools import partial
from typing import Dict

import calibration as cal
import optuna
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from scipy.optimize import minimize_scalar


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x) -> Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train_epoch(model, train_loader, optimizer, scheduler, loss_fn) -> None:
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        logit = model(data)
        logprob = F.log_softmax(logit, dim=1)
        loss = loss_fn(logprob, target)
        loss.backward()
        optimizer.step()
        scheduler.step()


@torch.no_grad()
def evaluate(model, test_loader) -> Dict[str, float]:
    model.eval()
    outputs = []
    targets = []
    for data, target in test_loader:
        output = model(data)
        outputs.append(output)
        targets.append(target)

    outputs = torch.cat(outputs)
    targets = torch.cat(targets)

    logprobs = F.log_softmax(outputs, dim=1)
    log_loss = F.nll_loss(logprobs, targets)
    accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()

    def temperature_error(log_temperature):
        temperature = math.exp(log_temperature)
        probs = F.softmax(outputs / temperature, dim=1)
        calibration_error = cal.get_calibration_error(probs, targets)
        if math.isnan(calibration_error):
            calibration_error = 9.0
        return calibration_error

    res = minimize_scalar(temperature_error)
    if res.success:
        temperature = math.exp(res.x)
    else:
        warnings.warn("temperature scaling search didn't converge")
        temperature = 1.0
    probs = F.softmax(outputs / temperature, dim=1)
    calibration_error = cal.get_calibration_error(probs, targets)
    ece = cal.get_ece(probs, targets)

    print(
        f"\n"
        f"Log loss: {log_loss:.4f}, "
        f"Accuracy: {100 * accuracy:.0f}%, "
        f"Temperature: {temperature:.3f}, "
        f"ECE: {ece:.4f}, "
        f"Calibration error: {calibration_error:.4f}"
        f"\n"
    )

    return {
        "accuracy": accuracy,
        "log_loss": log_loss,
        "calibration_error": calibration_error,
        "ece": ece,
        "temperature": temperature,
    }


def nll_loss_with_label_smoothing(
    logprob: Tensor, target: Tensor, *, smoothing: float = 0.0
) -> Tensor:
    nll_loss = -logprob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    smooth_loss = -logprob.mean(dim=-1)
    loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss
    return loss.mean()


def focal_loss(logprob: Tensor, target: Tensor, *, gamma: float = 1.0) -> Tensor:
    logprob = logprob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    loss = -(1 - logprob.exp()).pow(gamma) * logprob
    return loss.mean()


def load_datasets():
    normalize = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    official_train_dataset = MNIST(
        "data", train=True, download=True, transform=normalize
    )

    # split out a random validation set from official training dataset,
    # instead of doing hyperparam search on the official test set
    ids = list(range(len(official_train_dataset)))
    random.shuffle(ids)
    eval_size = int(0.1 * len(ids))
    train_dataset = Subset(official_train_dataset, ids[eval_size:])
    eval_dataset = Subset(official_train_dataset, ids[:eval_size])
    return {"train": train_dataset, "eval": eval_dataset}


def objective(trial) -> float:
    seed = random.randint(0, 0xFFFFFFFF)
    torch.manual_seed(seed)
    trial.set_user_attr("seed", seed)

    batch_size = 2 ** trial.suggest_int("log2_batch_size", 5, 11)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1.0, log=True)

    loss_name = trial.suggest_categorical(
        "loss_function", ["focal_loss", "nll_loss_with_label_smoothing"]
    )
    if loss_name == "nll_loss_with_label_smoothing":
        loss_fn = partial(
            nll_loss_with_label_smoothing,
            smoothing=trial.suggest_float("label_smoothing", 0.0, 1.0),
        )
    elif loss_name == "focal_loss":
        loss_fn = partial(
            focal_loss, gamma=trial.suggest_float("focal_loss_gamma", 1.0, 10.0)
        )
    else:
        raise ValueError(f"unknown loss: {loss_name}")

    datasets = load_datasets()
    train_loader = DataLoader(
        datasets["train"], batch_size=batch_size, shuffle=True, num_workers=1
    )
    eval_loader = DataLoader(datasets["eval"], batch_size=4096, num_workers=1)

    total_steps = len(train_loader)

    model = ConvNet()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)
    train_epoch(model, train_loader, optimizer, scheduler, loss_fn=loss_fn)
    metrics = evaluate(model, eval_loader)
    trial.set_user_attr("temperature", metrics["temperature"])

    # optimize calibration error but not at the expense of classification accuracy
    combined = -math.log(metrics["accuracy"]) + metrics["calibration_error"]

    # even though Optuna detects NaN's, the hyperparameter search sometimes gets stuck
    if math.isnan(combined):
        combined = 9.0
    return combined


def main():
    study = optuna.create_study(
        direction="minimize",
        study_name="mnist_calibration_tuning",
        storage="sqlite:///mnist_calibration_tuning.db",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=100)
    print(study.best_trial)


if __name__ == "__main__":
    main()

import copy
import os
import time

import torch
from torch import nn
from torch.optim import lr_scheduler
from torchvision import models

TRAIN_STEP = "train"
VALIDATION_STEP = "val"


def _loss(name):
    # TODO: Keeping it open for adding loss functions
    fn = None
    if name == "crossentropy":
        fn = torch.nn.CrossEntropyLoss()
    else:
        pass
    return fn


class CarNet(nn.Module):
    """The class represents the car classification neural network based on the RESNET18.
    
    Args:
        nn ([type]): [description]
    
    Returns:
        [type]: [description]
    """

    def __init__(
        self,
        num_classes,
        loss_fn="crossentropy",
        optimiser="adam",
        lr=0.001,
        pretrained=True,
        device=None,
        require_chkpoint=False,
        chkpnt_folder=None,
    ):
        # TODO: Add option for loading models which are trained.
        super().__init__()
        if device is not None:
            self.device = device
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model.to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=7, gamma=0.1)
        self.checkpoint = require_chkpoint
        self.chkpnt_folder = chkpnt_folder

    def forward(self, x):
        return self.model(x)

    def fit(self, train_dataloader, val_dataloader, epochs):
        start = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        for epoch in range(epochs):
            print(f"Running epoch {epoch + 1}/{epochs}")
            print("-" * 15)
            self._train_step(train_dataloader)
            _, epoch_acc = self._validation_step(val_dataloader)

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
        time_elapsed = time.time() - start
        print(
            "Training completed in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best Validation Accuracy: {:4f}".format(best_acc))
        # load best model weights
        self.model.load_state_dict(best_model_wts)
        if self.checkpoint:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.chkpnt_folder, "best_fit_model.pt"),
            )
        return self

    def _train_step(self, dataloader, epoch, verbose=True):
        start = time.time()
        self.scheduler.step()
        self.model.train()
        epoch_loss, epoch_acc = self._run(dataloader, TRAIN_STEP)
        time_elapsed = time.time() - start
        if self.checkpoint:
            self.save_checkpoint(epoch, "train_chk.pt")
        if verbose:
            print(
                "Completed in {:.0f}m {:.0f}s training loss: {:.4f}, accuracy: {:.4f}".format(
                    time_elapsed // 60, time_elapsed % 60, epoch_loss, epoch_acc
                )
            )
        return epoch_loss, epoch_acc

    def _run(self, dataloader, step):
        running_loss = 0.0
        running_corrects = 0.0

        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(self.device)
            batch_labels = batch_labels.to(self.device)

            self.optim.zero_grad()

            with torch.set_grad_enabled(step == TRAIN_STEP):
                outputs = self.forward(batch_inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.loss_fn(outputs, batch_labels)
                if step == TRAIN_STEP:
                    loss.backward()
                    self.optim.step()

            running_loss += loss.item() * batch_inputs.size(0)
            running_corrects += torch.sum(preds == batch_labels.data)

        epoch_loss, epoch_acc = (
            running_loss / len(dataloader.dataset),
            running_corrects.double() / len(dataloader.dataset),
        )
        return epoch_loss, epoch_acc

    def predict(self, X, k=1):
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            outputs = self.forward(X)
            probs = torch.exp(outputs)
        return probs.topk(k, dim=1)

    def evaluate(self, test_dataloader):
        self.model.eval()
        corrects = 0
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward(inputs)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
        return corrects.double() / len(test_dataloader.dataset) * 100

    def _validation_step(self, dataloader, epoch, step, verbose):
        start = time.time()
        self.model.eval()
        epoch_loss, epoch_acc = self._run(dataloader, VALIDATION_STEP)
        time_elapsed = time.time() - start
        if self.checkpoint:
            self.save_checkpoint(epoch, "val_chk.pt")
        if verbose:
            print(
                "Completed in {:.0f}m {:.0f}s validation loss: {:.4f}, accuracy: {:.4f}".format(
                    time_elapsed // 60, time_elapsed % 60, epoch_loss, epoch_acc
                )
            )
        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch, file):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
            },
            os.path.join(self.chkpnt_folder, file),
        )

    def load_checkpoint(self, file):
        checkpoint = torch.load(os.path.join(self.chkpnt_folder, file))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        return epoch

    def save_model(self, file):
        torch.save(self.model, file)

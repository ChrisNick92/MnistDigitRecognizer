import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.optim.lr_scheduler import StepLR
import os

def training_loop(model, train_loader, val_loader, epochs,
                  lr, loss_fn, regularization=None,
                  reg_lambda=None, mod_epochs=20, early_stopping = False,
                  patience = None, verbose = None, title = None,
                  scheduler_bool = False, gamma = 0.1, step_size = 10, model_name = "model"):
    optim = Adam(model.parameters(), lr=lr)
    if scheduler_bool:
        scheduler = StepLR(optim, step_size = step_size, gamma = gamma)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    train_loss_list = []
    val_loss_list = []
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    counter_epochs = 0

    if early_stopping:
        ear_stopping = EarlyStopping(patience= patience, verbose=verbose, path = model_name+".pt")

    for epoch in range(epochs):
        counter_epochs+=1
        model.train()
        train_loss, val_loss = 0.0, 0.0
        for train_batch in train_loader:
            X, y = train_batch[0].to(device), train_batch[1].to(device)
            preds = model(X)
            loss = loss_fn(preds, y)
            train_loss += loss.item()

            # Regulirization
            if regularization == 'L2':
                l_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + reg_lambda * l_norm
            elif regularization == 'L1':
                l_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + reg_lambda * l_norm

            # Backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                X, y = val_batch[0].to(device), val_batch[1].to(device)
                preds = model(X)
                val_loss += loss_fn(preds, y).item()
        train_loss /= num_train_batches
        val_loss /= num_val_batches
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        if (epoch + 1) % mod_epochs == 0:
            print(
                f"Epoch: {epoch + 1}/{epochs}{5 * ' '}Training Loss: {train_loss:.4f}{5 * ' '}Validation Loss: {val_loss:.4f}")

        if early_stopping:
            ear_stopping(val_loss, model)
            if ear_stopping.early_stop:
                print("Early stopping")
                break
        if scheduler_bool:
            scheduler.step()
    sns.set_style("dark")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, counter_epochs + 1), train_loss_list, label='Train Loss')
    ax.plot(range(1, counter_epochs + 1), val_loss_list, label='Val Loss')
    ax.set_title("Train - Val Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    plt.legend()
    plt.show()

    if early_stopping:
        model.load_state_dict(torch.load(os.path.join("modelcheckpoints", model_name+".pt")))



def test_loop(model, test_dloader, device='cpu'):
    predictions_list = np.array([], dtype=np.int64)
    targets_list = np.array([], dtype=np.int64)
    model.eval()

    for val_sample in test_dloader:
        X = val_sample[0].to(device)
        y = val_sample[1].cpu().numpy()
        targets_list = np.concatenate((targets_list, y))

        with torch.no_grad():
            preds = model(X)
            predictions_list = np.concatenate((predictions_list,
                                               torch.argmax(preds, dim=-1).cpu().numpy()))
    acc = len(targets_list[targets_list == predictions_list])/len(targets_list)
    return predictions_list, targets_list, acc



# Source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter % 5 == 0: # Print only per 5 degradations of val loss
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join("modelcheckpoints", self.path))
        self.val_loss_min = val_loss
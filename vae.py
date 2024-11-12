from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim.adam

from data_utils import load_mnist_data, _expand_int

@dataclass
class Hyperparameters:
    lr: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    num_hidden: int = 8


class AutoEncoder(nn.Module):
    def __init__(self, num_hidden: int = 8):
        super().__init__()
        self._mnist_input = 784
        self._num_hidden = num_hidden
        self.encoder = nn.Sequential(
            nn.Linear(self._mnist_input, 256),
            nn.ReLU(),
            nn.Linear(256, self._num_hidden),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self._num_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def __str__(self) -> str:
        return "autoencoder"

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def compute_loss(self, model_input: torch.Tensor):
        _, model_output = self.forward(model_input)
        loss = nn.functional.mse_loss(model_output, model_input, reduction="sum")
        return loss


class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, num_hidden: int = 8):
        super().__init__(num_hidden)
        self.mu = nn.Linear(self._num_hidden, self._num_hidden)
        self.logvar = nn.Linear(self._num_hidden, self._num_hidden)

    def __str__(self) -> str:
        return "vae"

    def reparam_trick(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        var = torch.exp(0.5 * logvar)
        eps = torch.randn_like(var)
        return mu + eps * var

    def forward(self, model_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(model_input)
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        dec_input = self.reparam_trick(mu, logvar)
        decoded = self.decoder(dec_input)
        return encoded, decoded, mu, logvar
    
    def compute_loss(self, model_input: torch.Tensor):
        _, model_output, mu, logvar = self.forward(model_input)
        recon_loss = nn.functional.mse_loss(model_output, model_input, reduction="sum")
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        return recon_loss + kl_loss

class CVAEMNIST(torch.utils.data.Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = _expand_int(labels)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]

class ConditionalVariationalAutoEncoder(VariationalAutoEncoder):
    def __init__(self, num_hidden: int = 8, num_classes: int = 10):
        super().__init__(num_hidden)
        self._num_classes = num_classes
        self.encode_label = nn.Sequential(
            nn.Linear(self._num_classes, self._num_hidden),
            nn.ReLU()
        )

    def __str__(self) -> str:
        return "cvae"
    
    def condition_on_label(self, label_vector: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        return encoded + self.encode_label(label_vector)

    def forward(self, model_input: torch.Tensor, input_label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(model_input)
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        reparamed_enc = self.reparam_trick(mu, logvar)
        dec_input = self.condition_on_label(input_label, reparamed_enc)
        decoded = self.decoder(dec_input)
        return encoded, decoded, mu, logvar
    
    def compute_loss(self, model_input: torch.Tensor, input_label: torch.Tensor):
        _, model_output, mu, logvar = self.forward(model_input, input_label)
        recon_loss = nn.functional.mse_loss(model_output, model_input, reduction="sum")
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        return recon_loss + 0.1 * kl_loss
    
EncTypes = Union[AutoEncoder, VariationalAutoEncoder, ConditionalVariationalAutoEncoder]

def train_loop(model_type: EncTypes):
    # Autoencoders are unsupervised, because we just need to recreate our input
    hp = Hyperparameters()
    images, _ = load_mnist_data(test = False)
    training_imgs = torch.from_numpy(images)

    model = model_type(hp.num_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr = hp.lr)
    device = torch.cuda.current_device()
    assert "GPU" in torch.cuda.get_device_name(device), "Not using GPU!"
    model.to(device)

    train_loader = torch.utils.data.DataLoader(
        training_imgs, batch_size=hp.batch_size, shuffle=True,
    )

    for epoch in range(hp.num_epochs):
        total_loss = 0.0
        for _, batch_data in enumerate(train_loader):
            data = batch_data.to(device)
            # Compare to original data
            loss = model.compute_loss(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} of {hp.num_epochs}: loss {epoch_loss}")
    torch.save(model.state_dict(), f"model/mnist_{model}.pt")


def train_cvae():
    # Autoencoders are unsupervised, because we just need to recreate our input
    hp = Hyperparameters()
    images, labels = load_mnist_data(test = False)
    training_imgs = torch.from_numpy(images)
    training_labels = torch.from_numpy(labels)

    model = ConditionalVariationalAutoEncoder(hp.num_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr = hp.lr)
    device = torch.cuda.current_device()
    assert "GPU" in torch.cuda.get_device_name(device), "Not using GPU!"
    model.to(device)

    dataset = CVAEMNIST(training_imgs, training_labels)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=hp.batch_size, shuffle=True,
    )

    for epoch in range(hp.num_epochs):
        total_loss = 0.0
        for batch_data, batch_labels in train_loader:
            data = batch_data.to(device)
            label_data = batch_labels.to(device)
            # Compare to original data
            loss = model.compute_loss(data, label_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} of {hp.num_epochs}: loss {epoch_loss}")
    torch.save(model.state_dict(), f"model/mnist_cvae.pt")


def view_model_output(model_filepath: str, model_type: EncTypes, num_samples: int = 6) -> None:
    model = model_type()
    model.load_state_dict(torch.load(model_filepath, weights_only=True))
    model.eval()
    # Test or view it
    images, labels = load_mnist_data(test = False)
    inds = np.random.randint(0, high = images.shape[0], size = num_samples)
    samples = images[inds, :]
    _, axs = plt.subplots(
        ncols = num_samples, nrows = 2, figsize = (10, 3 * num_samples)
    )
    for sample_idx in range(samples.shape[0]):
        sample_data = samples[sample_idx, :]
        sample_img = sample_data.reshape(28, 28)
        axs[0][sample_idx].imshow(sample_img, cmap="gray")
        axs[0][sample_idx].set_title(f"Labels: {labels[inds[sample_idx]]}")
        # Look at the outputs
        retvals = model(torch.from_numpy(sample_data))
        decoded_output = retvals[1]
        output_img_np = decoded_output.cpu().detach().numpy().reshape(28, 28)
        axs[1][sample_idx].imshow(output_img_np, cmap="gray")
        axs[1][sample_idx].set_title("Model Output")
    plt.show()

def view_cvae_output(model_filepath: str, input_vec = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])) -> None:
    model = ConditionalVariationalAutoEncoder()
    model.load_state_dict(torch.load(model_filepath, weights_only=True))
    model.eval()
    # Test or view it
    images, labels = load_mnist_data(test = False)
    inds = np.random.randint(0, high = images.shape[0], size = 10)
    samples = images[inds, :]
    _, axs = plt.subplots(
        ncols = 10, nrows = 2, figsize = (10, 3 * 10)
    )
    input_label_tensor = torch.from_numpy(input_vec.astype(np.float32))
    for sample_idx in range(samples.shape[0]):
        sample_data = samples[sample_idx, :]
        sample_img = sample_data.reshape(28, 28)
        axs[0][sample_idx].imshow(sample_img, cmap="gray")
        axs[0][sample_idx].set_title(f"Labels: {labels[inds[sample_idx]]}")
        # Look at the outputs
        retvals = model(torch.from_numpy(sample_data), input_label_tensor)
        decoded_output = retvals[1]
        output_img_np = decoded_output.cpu().detach().numpy().reshape(28, 28)
        axs[1][sample_idx].imshow(output_img_np, cmap="gray")
        axs[1][sample_idx].set_title("Conditioned on {input_vec}")
    plt.show()


def main():
    # train_loop(VariationalAutoEncoder)
    # view_model_output("model/mnist_vae.pt", VariationalAutoEncoder)
    # train_cvae()
    view_cvae_output("model/mnist_cvae.pt")

if __name__ == "__main__":
    main()


    



        







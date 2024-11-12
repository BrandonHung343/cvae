from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR: Path = Path.cwd() / "data"
MNIST_TRAIN: str = DATA_DIR / "mnist_train.npy"
MNIST_TEST: str = DATA_DIR / "mnist_test.npy"

def _expand_int(labels: np.ndarray) -> np.ndarray:
    one_hot = np.zeros((len(labels), 10))
    for i, label in enumerate(labels):
        one_hot[i, label] = 1
    return one_hot.astype(np.float32)

def load_save_data() -> None:
    filenames = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
                't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
    data = []

    for child in DATA_DIR.iterdir():
        filepath = str(child)
        if filepath[-3:] == "npy":
            continue
        with open(filepath, 'rb') as f:
            if 'labels' in filepath:
                # Load the labels as a one-dimensional array of integers
                # label_int = np.frombuffer(f.read(), np.uint8, offset=8)
                data.append(np.frombuffer(f.read(), np.uint8, offset=8))
            else:
                # Load the images as a two-dimensional array of pixels
                data.append(np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28))

    # Split into training and testing sets
    X_train, y_train, X_test, y_test = data

    # Normalize the pixel values
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Convert labels to integers
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    with open(str(MNIST_TRAIN), "wb") as f:
        np.save(f, X_train)
        np.save(f, y_train)

    with open(str(MNIST_TEST), "wb") as f:
        np.save(f, X_test)
        np.save(f, y_test)


def show_images(num_samples: int = 9, num_rows: int = 3, x_pix: int = 28, y_pix: int = 28) -> None:
    with open(str(MNIST_TRAIN), "rb") as f:
        images = np.load(f)
        labels = np.load(f)

    _, axs = plt.subplots(
        ncols = int(num_samples / num_rows), nrows = num_rows, figsize = (10, 3 * num_samples)
    )
    sample_inds = np.random.randint(low = 0, high = images.shape[0], size = num_samples)
    for i in range(num_samples):
        sample_idx = sample_inds[i]
        row_ind = int(i / num_rows)
        col_ind = int(i % num_rows)
        pixels = images[sample_idx, :].reshape(x_pix, y_pix)
        axs[row_ind][col_ind].imshow(pixels, cmap="gray")
        axs[row_ind][col_ind].set_title(f"Labels: {labels[sample_idx]}")
    plt.show()
    

def load_mnist_data(test: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    data_file = MNIST_TRAIN
    if test:
        data_file = MNIST_TEST
    with open(str(data_file), "rb") as f:
        images = np.load(f)
        labels = np.load(f)
    return images, labels


def show_given_images(
    images_list: np.ndarray, num_samples: int = 9, num_rows: int = 3, x_pix: int = 28, y_pix: int = 28
) -> None:
    _, axs = plt.subplots(
        ncols = int(num_samples / num_rows), nrows = num_rows, figsize = (10, 3 * num_samples)
    )
    sample_inds = np.random.randint(low = 0, high = images_list[0].shape[0], size = num_samples)
    for i in range(num_samples):
        sample_idx = sample_inds[i]
        row_ind = int(i / num_rows)
        col_ind = int(i % num_rows)
        pixels = images[sample_idx, :].reshape(x_pix, y_pix)
        axs[row_ind][col_ind].imshow(pixels, cmap="gray")
        axs[row_ind][col_ind].set_title(f"Labels: {labels[sample_idx]}")
    plt.show()
    


if __name__ == "__main__":
    # load_save_data()
    show_images()
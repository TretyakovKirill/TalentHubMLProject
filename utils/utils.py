import os
import scipy
import torch
import numpy as np

from typing import Literal, Tuple


def load_data(
        dir_path: str,
        dataset_type: Literal['A', 'B'],
        subject_id: int,
        mode: Literal['train', 'test'] = 'train'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load train or test dataset for a specific subject.

    :param dir_path: Directory where the data is stored.
    :param dataset_type: "A" for BCI IV-2a, "B" for BCI IV-2b.
    :param subject_id: Subject number (1 to 9).
    :param mode: "train" or "test".
    :return: Tuple of (data, labels).
    """
    suffix = 'T' if mode == 'train' else 'E'
    filename = os.path.join(dir_path, f"{dataset_type}{subject_id:02d}{suffix}.mat")
    mat_data = scipy.io.loadmat(filename)
    data = mat_data['data']
    label = mat_data['label']
    return data, label


def prepare_dataset(
        dir_path: str,
        dataset_type: Literal['A', 'B'],
        subject_id: int,
        evaluation_mode: Literal['LOSO', 'subject-dependent'] = 'LOSO'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset according to evaluation mode.

    :param dir_path: Directory where the data is stored.
    :param dataset_type: "A" or "B".
    :param subject_id: Subject number (1 to 9).
    :param evaluation_mode: "LOSO" for cross-subject evaluation, any other value for subject-dependent evaluation.
    :return: (X_train, y_train, X_test, y_test)
    """
    if not 1 <= subject_id <= 9:
        raise ValueError(f"subject_id must be between 1 and 9, got {subject_id}")

    if evaluation_mode == 'LOSO':
        X_train, y_train = [], []
        X_test, y_test = None, None

        for current_subject_id in range (1, 10):
            X_train_part, y_train_part = load_data(dir_path, dataset_type, current_subject_id, mode='train')
            X_test_part, y_test_part = load_data(dir_path, dataset_type, current_subject_id, mode='test')

            X = np.concatenate((X_train_part, X_test_part), axis=0)
            y = np.concatenate((y_train_part, y_test_part), axis=0)

            if current_subject_id == subject_id:
                X_test, y_test = X, y
            else:
                X_train.append(X)
                y_train.append(y)

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        return X_train, y_train, X_test, y_test
    else:
        X_train, y_train = load_data(dir_path, dataset_type, subject_id, mode='train')
        X_test, y_test = load_data(dir_path, dataset_type, subject_id, mode='test')

        return X_train, y_train, X_test, y_test


def sr_augmentation(
    signals: np.ndarray,
    labels: np.ndarray,
    num_augmentations: int,
    batch_size: int,
    num_classes: int,
    num_segments: int,
    num_channels: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Segmentation and Reconstruction (S&R) data augmentation.

    :param signals: Input EEG data array of shape [B, 1, num_channels, 1000].
    :param labels: Labels array corresponding to signals.
    :num_augmentations: Number of augmentations per class.
    :batch_size: Batch size.
    :param num_classes: Number of classes in the dataset.
    :param num_segments: Number of segments to divide each trial into.
    :param device: Device to move augmented tensors to ("cuda" or "cpu").
    :return: Tuple of (augmented_data, augmented_labels) as torch Tensors.
    """
    num_examples = batch_size // num_classes
    synth_per_class = num_augmentations * num_examples
    segment_length = signals.shape[-1] // num_segments

    synth_signals = []
    synth_labels = []

    for class_id in range(1, num_classes + 1):
        mask = (labels == class_id)

        class_signals = signals[mask]
        class_lbls = labels[mask]

        random_indices = np.random.randint(0, class_signals.shape[0], (synth_per_class, num_segments))

        synth_signals_cls = np.zeros(
            (synth_per_class, 1, num_channels, signals.shape[-1]),
            dtype=class_signals.dtype
        )

        for seg_id in range(num_segments):
            start = seg_id * segment_length
            end   = start + segment_length
            picks = random_indices[:, seg_id]
            synth_signals_cls[:, :, :, start:end] = class_signals[picks, :, :, start:end]

        synth_labels_cls = np.full(
            (synth_per_class,),
            class_id,
            dtype=labels.dtype
        )

        synth_signals.append(synth_signals_cls)
        synth_labels.append(synth_labels_cls)

    augmented_signals = np.concatenate(synth_signals, axis=0)
    augmented_labels  = np.concatenate(synth_labels, axis=0) - 1

    perm = np.random.permutation(augmented_signals.shape[0])
    augmented_signals = augmented_signals[perm]
    augmented_labels  = augmented_labels[perm]

    augmented_signals = torch.tensor(augmented_signals, dtype=torch.float32, device=device)
    augmented_labels  = torch.tensor(augmented_labels, dtype=torch.long, device=device)

    return augmented_signals, augmented_labels

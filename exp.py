"""
За основу класса обучения и тестирования взят класс ExP, реализованный в проекте EEG Conformer: https://github.com/eeyhsong/EEG-Conformer
"""


import os
import time
import random

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from utils import prepare_dataset, sr_augmentation
from model import Model


class ExP():
    def __init__(
        self,
        subject_id,
        data_dir,
        result_name,
        n_epochs=20,
        n_aug=2,
        n_seg=8,
        validate_ratio=0.2,
        learning_rate=0.001,
        batch_size=200,
        opt_b1=0.5,
        opt_b2=0.999,
        evaluate_mode='subject-dependent',
        dataset_type='A',
        n_class=4,
        n_channel=22,
        emb_size=40,
        transformer_heads=4,
        transformer_depth=6,
        cnn_f1=20,
        cnn_kernel_size=64,
        cnn_expansion_factor=2,
        cnn_pooling_size1 = 8,
        cnn_pooling_size2 = 8,
        cnn_dropout_rate = 0.3,
        cnn_flatten = 600,
        device='cpu'
    ):
        super(ExP, self).__init__()
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.lr = learning_rate
        self.opt_b1 = opt_b1
        self.opt_b2 = opt_b2
        self.n_epochs = n_epochs
        self.subject_id = subject_id
        self.n_aug = n_aug
        self.n_seg = n_seg
        self.data_dir = data_dir
        self.transformer_heads=transformer_heads
        self.emb_size=emb_size
        self.transformer_depth=transformer_depth
        self.result_name = result_name
        self.evaluate_mode = evaluate_mode
        self.validate_ratio = validate_ratio
        self.device = device

        self.n_class = n_class
        self.n_channel = n_channel

        self.criterion_cls = nn.CrossEntropyLoss().to(self.device)

        self.model = Model(
            emb_size=self.emb_size,
            heads=self.transformer_heads,
            depth=self.transformer_depth,
            cnn_f1=cnn_f1,
            cnn_expansion_factor=cnn_expansion_factor,
            cnn_kernel_size=cnn_kernel_size,
            cnn_pooling_size1=cnn_pooling_size1,
            cnn_pooling_size2=cnn_pooling_size2,
            cnn_dropout_rate=cnn_dropout_rate,
        ).to(self.device)

        self.model_filename = self.result_name + '/' + f'Model_{self.subject_id}.pth'

    def get_source_data(self):
        (train_data,
         train_label,
         test_data,
         test_label) = prepare_dataset(self.data_dir, self.dataset_type, self.subject_id, self.evaluate_mode)

        train_data = np.expand_dims(train_data, axis=1)
        train_label = np.transpose(train_label)

        allData = train_data
        allLabel = train_label[0]

        shuffle_num = np.random.permutation(len(allData))
        allData = allData[shuffle_num, :, :, :]
        allLabel = allLabel[shuffle_num]


        print('-'*20, "train size:", train_data.shape, "test size:", test_data.shape)
        test_data = np.expand_dims(test_data, axis=1)
        test_label = np.transpose(test_label)

        testData = test_data
        testLabel = test_label[0]

        target_mean = np.mean(allData)
        target_std = np.std(allData)
        allData = (allData - target_mean) / target_std
        testData = (testData - target_mean) / target_std

        return allData, allLabel, testData, testLabel


    def train(self):
        raw_train_data, raw_train_label, test_data, test_label = self.get_source_data()

        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(raw_train_data),
            torch.from_numpy(raw_train_label - 1)
        )

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.opt_b1, self.opt_b2))

        test_data = test_data.to(self.device, dtype=torch.float32)
        test_label = test_label.to(self.device, dtype=torch.long)

        best_epoch = 0
        num = 0
        min_loss = 100

        result_process = []

        for e in range(self.n_epochs):
            self.train_dataloader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )

            epoch_process = {}
            epoch_process['epoch'] = e
            self.model.train()
            outputs_list = []
            label_list = []

            val_data_list = []
            val_label_list = []

            for i, (batch_data, batch_labels) in enumerate(self.train_dataloader):
                number_sample = batch_data.shape[0]
                number_validate = int(self.validate_ratio * number_sample)

                train_data = batch_data[:-number_validate]
                train_label = batch_labels[:-number_validate]
                val_data = batch_data[number_validate:]
                val_label = batch_labels[number_validate:]

                val_data_list.append(val_data)
                val_label_list.append(val_label)

                train_data = train_data.to(self.device, dtype=torch.float32)
                train_label = train_label.to(self.device, dtype=torch.long)

                aug_data, aug_label = sr_augmentation(
                    raw_train_data,
                    raw_train_label,
                    self.n_aug,
                    self.batch_size,
                    self.n_class,
                    self.n_seg,
                    self.n_channel,
                    self.device
                )

                train_data = torch.cat((train_data, aug_data))
                train_label = torch.cat((train_label, aug_label))

                features, outputs = self.model(train_data)
                outputs_list.append(outputs)
                label_list.append(train_label)

                loss = self.criterion_cls(outputs, train_label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (e + 1) % 10 == 0:
                self.model.eval()
                val_data = torch.cat(val_data_list).to(self.device, dtype=torch.float32)
                val_label = torch.cat(val_label_list).to(self.device, dtype=torch.long)

                val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
                self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)
                outputs_list = []
                with torch.no_grad():
                    for i, (train_data, _) in enumerate(self.val_dataloader):
                        train_data = train_data.to(self.device, dtype=torch.float32)
                        _, Cls = self.model(train_data)
                        outputs_list.append(Cls)
                        del train_data, Cls
                        torch.cuda.empty_cache()

                Cls = torch.cat(outputs_list)

                val_loss = self.criterion_cls(Cls, val_label)
                val_pred = torch.max(Cls, 1)[1]
                val_acc = float((val_pred == val_label).cpu().numpy().astype(int).sum()) / float(val_label.size(0))

                epoch_process['val_acc'] = val_acc
                epoch_process['val_loss'] = val_loss.detach().cpu().numpy()

                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == train_label).cpu().numpy().astype(int).sum()) / float(train_label.size(0))
                epoch_process['train_acc'] = train_acc
                epoch_process['train_loss'] = loss.detach().cpu().numpy()

                num = num + 1

                if val_loss < min_loss:
                    min_loss = val_loss
                    best_epoch = e
                    epoch_process['epoch'] = e
                    torch.save(self.model.state_dict(), self.model_filename)

                print("{}_{} train_acc: {:.4f} train_loss: {:.6f} val_acc: {:.6f} val_loss: {:.7f}".format(
                    self.subject_id,
                    epoch_process['epoch'],
                    epoch_process['train_acc'],
                    epoch_process['train_loss'],
                    epoch_process['val_acc'],
                    epoch_process['val_loss'],
                ))

            result_process.append(epoch_process)

            if (e + 1) % 50 == 0:
                self.model.eval()
                test_outputs = []
                test_labels = []
                with torch.no_grad():
                    for x, y in self.test_dataloader:
                        x = x.to(self.device, dtype=torch.float32)
                        y = y.to(self.device, dtype=torch.long)
                        _, out = self.model(x)
                        test_outputs.append(out)
                        test_labels.append(y)
                        del x, y, out
                        torch.cuda.empty_cache()

                test_outputs = torch.cat(test_outputs)
                test_labels = torch.cat(test_labels)
                test_pred = test_outputs.argmax(dim=1)
                test_acc = (test_pred == test_labels).float().mean().item()

                print(f"=== Epoch {e+1} — test accuracy: {test_acc:.4f} ===")
                
                save_path = f"{self.model_filename}_e{e}.pth"
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved in {save_path}")

        self.model.eval()
        state_dict = torch.load(self.model_filename, map_location=self.device)
        self.model.load_state_dict(state_dict)
        outputs_list = []
        with torch.no_grad():
            for test_data, label in self.test_dataloader:
                test_data = test_data.to(self.device, dtype=torch.float32)

                features, outputs = self.model(test_data)
                val_pred = torch.max(outputs, 1)[1]
                outputs_list.append(outputs)

        outputs = torch.cat(outputs_list)
        y_pred = torch.max(outputs, 1)[1]

        test_acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))

        print("Epoch:", best_epoch, 'Test accuracy is:', test_acc)

        df_process = pd.DataFrame(result_process)

        return test_acc, test_label, y_pred, df_process, best_epoch


if __name == "__main__":
    seed_n = random.randint(0, 2048)
    print('Seed:', seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    DATA_DIR = './data/'
    EVALUATE_MODE = 'subject-dependent' # (LOSO / subject-dependent)
    N_EPOCHS = 600
    LEARNING_RATE = 0.001
    BATCH_SIZE = 250

    N_SUBJECT = 9
    N_AUG = 3
    N_SEG = 8

    DATASET_TYPE = 'A'
    N_CLASS = 4 if DATASET_TYPE == 'A' else 2
    N_CHANNEL = 22 if DATASET_TYPE == 'A' else 3

    VALIDATE_RATIO = 0.3

    EMB_DIM = 16
    TRANSFORMER_HEADS = 2
    TRANSFORMER_DEPTH = 3

    CNN_F1 = 8
    CNN_KERNEL_SIZE = 64
    CNN_EXPANSION_FACTOR = 2
    CNN_POOL_SIZE1 = 8
    CNN_POOL_SIZE2 = 8
    CNN_FLATTEN = 144
    CNN_DROPOUT_RATE = 0.25 if EVALUATE_MODE == 'LOSO' else 0.5

    RESULT_NAME = f"Model_{DATASET_TYPE}_heads_{TRANSFORMER_HEADS}_depth_{TRANSFORMER_DEPTH}_{int(time.time())}"
    if not os.path.exists(RESULT_NAME):
        os.makedirs(RESULT_NAME)

    subjects_result = []
    best_epochs = []

    for subject_id in range(N_SUBJECT):
        print(f'Subject {subject_id + 1}')

        exp = ExP(
            subject_id=subject_id + 1,
            data_dir=DATA_DIR,
            result_name=RESULT_NAME,
            n_epochs=N_EPOCHS,
            n_aug=N_AUG,
            n_seg=N_SEG,
            evaluate_mode=EVALUATE_MODE,
            emb_size=EMB_DIM,
            transformer_heads=TRANSFORMER_HEADS,
            transformer_depth=TRANSFORMER_DEPTH,
            dataset_type=DATASET_TYPE,
            n_class=N_CLASS,
            n_channel=N_CHANNEL,
            cnn_f1=CNN_F1,
            cnn_kernel_size=CNN_KERNEL_SIZE,
            cnn_expansion_factor=CNN_EXPANSION_FACTOR,
            cnn_pooling_size1=CNN_POOL_SIZE1,
            cnn_pooling_size2=CNN_POOL_SIZE2,
            cnn_dropout_rate=CNN_DROPOUT_RATE,
            cnn_flatten=CNN_FLATTEN,
            validate_ratio=VALIDATE_RATIO,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            device=device
        )

        test_acc, Y_true, Y_pred, df_process, best_epoch = exp.train()
        best_epochs.append(best_epoch)

        true_np = Y_true.cpu().numpy().astype(int)
        pred_np = Y_pred.cpu().numpy().astype(int)

        accuracy = accuracy_score(true_np, pred_np)
        kappa = cohen_kappa_score(true_np, pred_np)

        print(
        f"""
        Subject {subject_id + 1} Results
        ---------------------------
        Accuracy : {100*accuracy:.2f} %
        Kappa    : {100*kappa:.2f}
        """
        )

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tft_model import TFT, QuantileLoss
from data_formatters import ts_dataset
from data_formatters.ts_dataset import TSDataset
# import data_formatters.base
from expt_settings.configs import ExperimentConfig
import importlib
from data_formatters import utils
import torch.optim as optim
from torch import device
from torch import Tensor
from pandas import DataFrame
from typing import List, Dict
import argparse
from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os


class Trainer:
    def __init__(self, model: TFT, loss: QuantileLoss, optimizer: optim.Adam, exp_config: ExperimentConfig):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device: device = model.device
        self.train_losses = []
        self.val_losses = []
        self.counter: int = 0
        # self.mse_val: float = 0
        self.min_val_mse: float = 9999
        self.exp_config = exp_config

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        n_iter: int = 0
        self.model.train()
        for i in range(epochs):
            epoch_loss = []
            # j = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                output, encoder_output, decoder_output, attn, attn_weights, emb_enc, emb_dec = self.model(batch)
                targets: Tensor = batch['outputs'].to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                loss: Tensor = self.loss(output[:, :, :].view(-1, 3), targets[:, :, 0].flatten().float())
                n_iter += 1
                loss.backward()
                epoch_loss.append(loss.item())
                self.optimizer.step()
                # j += 1

                # # update lr every 10000 iterations
                # if n_iter % update_lr == 0 and n_iter != 0:
                #     print("updating lr")
                #     for param_group in self.optimizer.param_groups:
                #         param_group['lr'] = param_group['lr'] * 0.9
            self.train_losses.append(np.mean(epoch_loss))
            print(np.mean(epoch_loss))

            self.eval(val_loader)
            self.save_checkpoint()
            if self.counter == self.patience:
                break

    def eval(self, val_loader: DataLoader):
        with torch.no_grad():
            loss_val: List = []
            for val_batch in val_loader:
                output, encoder_ouput, decoder_output, attn, attn_weights, emb_enc, emb_dec = self.model(val_batch)
                val_targets = val_batch['outputs'].to(self.device)
                loss: Tensor = self.loss(output[:, :, :].view(-1, 3), val_targets[:, :, 0].flatten().float())
                loss_val.append(loss.item())
            self.val_losses.append(np.mean(loss_val))
            # print(np.mean(loss_val))
        self.mse_val = np.mean(loss_val)

    def save_checkpoint(self):
        if self.mse_val < self.min_val_mse:
            self.min_val_mae: float = self.mse_val
            print("Saving...")
            print(self.exp_config.model_folder)
            torch.save(self.model.state_dict(),
                       os.path.join(self.exp_config.model_folder,
                                    f"TemporalFusionTransformer_{self.exp_config.experiment}.pt"))
            self.counter = 0
        else:
            self.counter += 1

    def plot_metrics(self):
        pass

    def plot_losses(self):
        fig1: Figure = plt.figure()
        plt.semilogy(range(len(self.train_losses)), self.train_losses)
        plt.title("train Loss per epoch")
        plt.show()
        plt.close(fig1)

        # plot validation losses
        fig2: Figure = plt.figure()
        plt.semilogy(range(len(self.val_losses)), self.val_losses)
        plt.title("validation Loss per epoch")
        plt.show()
        plt.close(fig2)


def main(exp_name: str, data_csv_path: str):
    exp_config = ExperimentConfig(exp_name, 'outputs')
    data_formatter = exp_config.make_data_formatter()
    print("*** Training from defined parameters for {} ***".format(exp_name))
    print("Loading & splitting data...")
    raw_data: DataFrame = pd.read_csv(data_csv_path, index_col=0)
    train, valid, test = data_formatter.split_data(raw_data)
    # train_samples, valid_samples = data_formatter.get_num_samples_for_calibration(
    # )
    # Sets up default params
    fixed_params: Dict = data_formatter.get_experiment_params()
    params: Dict = data_formatter.get_default_model_params()
    # TODO set the following in a proper config object
    id_col = 'id'
    time_col = 'hours_from_start'
    input_cols = ['power_usage', 'hour', 'day_of_week', 'hours_from_start', 'categorical_id']
    target_col = 'power_usage'
    static_cols = ['categorical_id']
    time_steps = 192
    num_encoder_steps = 168
    output_size = 1
    max_samples = 1000
    input_size = 5
    elect_train: TSDataset = ts_dataset.TSDataset(id_col, static_cols, time_col, input_cols,
                                                  target_col, time_steps, 10000,
                                                  input_size, num_encoder_steps, 1, output_size, train)
    elect_valid: TSDataset = ts_dataset.TSDataset(id_col, static_cols, time_col, input_cols,
                                                  target_col, time_steps, 1000,
                                                  input_size, num_encoder_steps, 1, output_size, valid)
    batch_size = 64
    train_loader = DataLoader(
        elect_train,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False
    )
    valid_loader = DataLoader(
        elect_valid,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False
    )
    # for batch in loader:
    #     break
    static_cols = ['meter']
    categorical_cols = ['hour']
    real_cols: List = ['power_usage', 'hour', 'day']
    config = {}
    config['static_variables'] = len(static_cols)
    config['time_varying_categoical_variables'] = 1
    config['time_varying_real_variables_encoder'] = 4
    config['time_varying_real_variables_decoder'] = 3
    config['num_masked_series'] = 1
    config['static_embedding_vocab_sizes'] = [369]
    config['time_varying_embedding_vocab_sizes'] = [369]
    config['embedding_dim'] = 8
    config['lstm_hidden_dimension'] = 160
    config['lstm_layers'] = 1
    config['dropout'] = 0.05
    config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config['batch_size'] = 64
    config['encode_length'] = 168
    config['attn_heads'] = 4
    config['num_quantiles'] = 3
    config['vailid_quantiles'] = [0.1, 0.5, 0.9]
    config['seq_length'] = 192
    config['static_variables'] = len(static_cols)
    print(f"Using {config['device']}")
    # instantiate model
    model: TFT = TFT(config)
    # # do a forward pass
    # output, encoder_output, decoder_output, \
    # attn, attn_output_weights, embeddings_encoder, embeddings_decoder = model.forward(batch)
    # define loss
    q_loss_func: QuantileLoss = QuantileLoss([0.1, 0.5, 0.9])
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # start training cycle
    # instantiate trainer
    trainer = Trainer(model=model, loss=q_loss_func, optimizer=optimizer, exp_config=exp_config)
    trainer.train(train_loader=train_loader, val_loader=valid_loader, epochs=100)
    trainer.plot_losses()
    # model.train()
    # epochs = 10
    # losses = []
    # for i in range(epochs):
    #     epoch_loss = []
    #     j = 0
    #     for batch in loader:
    #         output, encoder_ouput, decoder_output, attn, attn_weights, emb_enc, emb_dec = model(batch)
    #         loss: Tensor = q_loss_func(output[:, :, :].view(-1, 3), batch['outputs'][:, :, 0].flatten().float())
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss.append(loss.item())
    #         j += 1
    #         if j > 5:
    #             break
    #     losses.append(np.mean(epoch_loss))
    #     print(np.mean(epoch_loss))
    #
    # output, encoder_ouput, decoder_output, attn, attn_weights, emb_enc, emb_dec = model(batch)
    # ind = np.random.choice(64)
    # print(ind)
    # plt.plot(output[ind, :, 0].detach().cpu().numpy(), label='pred_1')
    # plt.plot(output[ind, :, 1].detach().cpu().numpy(), label='pred_5')
    # plt.plot(output[ind, :, 2].detach().cpu().numpy(), label='pred_9')
    #
    # plt.plot(batch['outputs'][ind, :, 0], label='true')
    # plt.legend()
    # plt.matshow(attn_weights.detach().numpy()[0, :, :])
    # plt.imshow(attn_weights.detach().numpy()[0, :, :])


if __name__ == "__main__":
    parser: ArgumentParser = argparse.ArgumentParser(description="Data download configs")
    parser.add_argument(
        "exp_name",
        metavar="e",
        type=str,
        nargs="?",
        default="electricity",
        help="Experiment Name"
    )
    parser.add_argument(
        "data_csv_path",
        metavar="f",
        type=str,
        nargs="?",
        default="data/hourly_electricity.csv",
        help="Path to folder for data download"
    )
    args: Namespace = parser.parse_args()
    main(args.exp_name, args.data_csv_path)

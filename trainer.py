import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tft_model import TFT, QuantileLoss
from data_formatters import ts_dataset
from data_formatters.ts_dataset import TSDataset
# import data_formatters.base
from expt_settings.configs import ExperimentConfig, config
import importlib
from data_formatters import utils
import torch.optim as optim
from torch import Tensor
from pandas import DataFrame
from typing import List, Dict
import argparse
from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt


def main(exp_name: str, data_csv_path: str):
    config = ExperimentConfig(exp_name, 'outputs')
    data_formatter = config.make_data_formatter()
    print("*** Training from defined parameters for {} ***".format('electricity'))
    print("Loading & splitting data...")
    raw_data: DataFrame = pd.read_csv(data_csv_path, index_col=0)
    train, valid, test = data_formatter.split_data(raw_data)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration(
    )
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
    elect: TSDataset = ts_dataset.TSDataset(id_col, static_cols, time_col, input_cols,
                                            target_col, time_steps, max_samples,
                                            input_size, num_encoder_steps, 1, output_size, train)
    batch_size = 64
    loader = DataLoader(
        elect,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True
    )
    for batch in loader:
        break
    static_cols = ['meter']
    categorical_cols = ['hour']
    real_cols: List = ['power_usage', 'hour', 'day']
    config['static_variables'] = len(static_cols)
    print(f"Using {config['device']}")
    # instantiate model
    model: TFT = TFT(config)
    # do a forward pass
    output, encoder_output, decoder_output, \
    attn, attn_output_weights, embeddings_encoder, embeddings_decoder = model.forward(batch)
    # define loss
    q_loss_func: QuantileLoss = QuantileLoss([0.1, 0.5, 0.9])
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # start training cycle
    model.train()
    epochs = 10
    losses = []
    for i in range(epochs):
        epoch_loss = []
        j = 0
        for batch in loader:
            output, encoder_ouput, decoder_output, attn, attn_weights, emb_enc, emb_dec = model(batch)
            loss: Tensor = q_loss_func(output[:, :, :].view(-1, 3), batch['outputs'][:, :, 0].flatten().float())
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            j += 1
            if j > 5:
                break
        losses.append(np.mean(epoch_loss))
        print(np.mean(epoch_loss))

    output, encoder_ouput, decoder_output, attn, attn_weights, emb_enc, emb_dec = model(batch)
    ind = np.random.choice(64)
    print(ind)
    plt.plot(output[ind, :, 0].detach().cpu().numpy(), label='pred_1')
    plt.plot(output[ind, :, 1].detach().cpu().numpy(), label='pred_5')
    plt.plot(output[ind, :, 2].detach().cpu().numpy(), label='pred_9')

    plt.plot(batch['outputs'][ind, :, 0], label='true')
    plt.legend()
    plt.matshow(attn_weights.detach().numpy()[0, :, :])
    plt.imshow(attn_weights.detach().numpy()[0, :, :])


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

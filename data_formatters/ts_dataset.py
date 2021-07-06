import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict
from pandas import DataFrame
from numpy import ndarray
from collections import OrderedDict


class TSDataset(Dataset):
    ## Mostly adapted from original TFT Github, data_formatters
    def __init__(self, id_col: str, static_cols: List, time_col: str, input_cols: List[str],
                 target_col: str, time_steps: int, max_samples: int,
                 input_size: int, num_encoder_steps: int, num_static: int,
                 output_size: int, data: DataFrame):

        self.time_steps = time_steps
        self.input_size = input_size
        self.output_size = output_size
        self.num_encoder_steps = num_encoder_steps

        data.sort_values(by=[id_col, time_col], inplace=True)
        print('Getting valid sampling locations.')

        valid_sampling_locations: List = []
        # split_data_map: Dict = {}
        split_data_map: OrderedDict = OrderedDict()
        for identifier, df in data.groupby(id_col):
            num_entries: int = len(df)
            if num_entries >= self.time_steps:
                valid_sampling_locations += [
                    (identifier, self.time_steps + i)
                    for i in range(num_entries - self.time_steps + 1)
                ]
            split_data_map[identifier]: DataFrame = df

        self.inputs: ndarray = np.zeros((max_samples, self.time_steps, self.input_size))
        self.outputs: ndarray = np.zeros((max_samples, self.time_steps, self.output_size))
        self.time: ndarray = np.empty((max_samples, self.time_steps, 1))
        self.identifiers: ndarray = np.empty((max_samples, self.time_steps, num_static))

        if 0 < max_samples < len(valid_sampling_locations):
            print('Extracting {} samples...'.format(max_samples))
            ranges: List = [valid_sampling_locations[i] for i in np.random.choice(
                len(valid_sampling_locations), max_samples, replace=False)]
        else:
            print('Max samples={} exceeds # available segments={}'.format(
                max_samples, len(valid_sampling_locations)))
            ranges: List = valid_sampling_locations

        for i, tup in enumerate(ranges):
            if ((i + 1) % 10000) == 0:
                print(i + 1, 'of', max_samples, 'samples done...')
            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx -
                                                     self.time_steps:start_idx]
            self.inputs[i, :, :] = sliced[input_cols]
            self.outputs[i, :, :] = sliced[[target_col]]
            self.time[i, :, 0] = sliced[time_col]
            self.identifiers[i, :, :] = sliced[static_cols]

        self.sampled_data = {
            'inputs': self.inputs,
            'outputs': self.outputs[:, self.num_encoder_steps:, :],
            'active_entries': np.ones_like(self.outputs[:, self.num_encoder_steps:, :]),
            'time': self.time,
            'identifier': self.identifiers
        }

    def __getitem__(self, index) -> Dict:
        s = {
            'inputs': self.inputs[index],
            'outputs': self.outputs[index, self.num_encoder_steps:, :],
            'active_entries': np.ones_like(self.outputs[index, self.num_encoder_steps:, :]),
            'time': self.time[index],
            'identifier': self.identifiers[index]
        }

        return s

    def __len__(self) -> int:
        return self.inputs.shape[0]

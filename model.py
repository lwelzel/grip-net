from typing import Any

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from scipy.interpolate import make_interp_spline
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import lightning as L

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')
torch.set_default_dtype(torch.float32)


data_csv = Path("./data/normalized_data.csv")
# data_csv = Path("./data/full_data.csv")


class CustomDataSet(Dataset):
    def __init__(self, csv_file, start=0., end=1., transform=True, validate=False, correct_imbalance=False):
        super(CustomDataSet, self).__init__()
        self.correct_imbalance = correct_imbalance
        self.image_size = 224.
        self.point_size = torch.tensor([5., 8., 10.])  # RGB

        self.df = pd.read_csv(csv_file)

        start_idx = int(self.df.shape[0] * start)
        end_idx = int(self.df.shape[0] * end)


        self.training_input_data = torch.tensor(self.df.to_numpy()[start_idx:end_idx, 2:], dtype=torch.float32)
        # TODO: fix this
        uncert_scalar = 1e-5
        self.preproc_error_std = torch.tensor([0.5, 0.5, 0.25]).view(1, 1, 3) * uncert_scalar

        # self.normalize(save=True)

        self.ground_truth = torch.tensor(
            self.df["ground_truth_safety_margin"].to_numpy()[start_idx:end_idx],
            dtype=torch.float32
        ).unsqueeze(1)

        self.weights = self.equalize_sample_weights(self.ground_truth)

    def __len__(self):
        return self.training_input_data.shape[0]

    def __getitem__(self, index):
        noisy_data = self.transform_add_noise(self.training_input_data[index])
        return noisy_data, self.ground_truth[index]

    def equalize_sample_weights(self, ground_truth, verbose=False):
        if not self.correct_imbalance:
            return torch.ones_like(ground_truth)

        density, bin_edges = torch.histogram(ground_truth, bins=50, range=(0., 1.), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        spline = make_interp_spline(bin_centers.cpu().numpy(), density.cpu().numpy(), k=1)
        weights = spline(ground_truth.cpu().numpy())
        weights = 1. / torch.tensor(weights, dtype=torch.float32).flatten()
        weights = torch.clip(weights, 1e-4, 1e2)

        if verbose:
            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.histplot(ground_truth, bins=100, kde=True)
            plt.show()

            sns.scatterplot(x=bin_edges[:-1], y=density)
            plt.show()

            sns.scatterplot(x=ground_truth.flatten(), y=weights.flatten())
            plt.show()

        return weights

    def normalize(self, save=False):
        self.training_input_data = self.training_input_data.view(self.training_input_data.shape[0], -1, 3, 3)

        radii = torch.square(self.training_input_data[:, :, :, :2]).sum(dim=-1).sqrt()
        mean_radius = torch.mean(radii, dim=0)
        self.training_input_data[:, :, :, :2] = self.training_input_data[:, :, :, :2] / mean_radius.view(1, -1, 3, 1)

        self.training_input_data[:, :, :, -1] = self.training_input_data[:, :, :, -1] / self.point_size

        self.training_input_data = self.training_input_data.view(self.training_input_data.shape[0], -1)

        # TODO: not div by std
        # std = torch.std(self.training_input_data, dim=0)
        # self.training_input_data = self.training_input_data / std

        if save:
            out = self.training_input_data.detach().clone().cpu().numpy()

            df_input = pd.DataFrame(out, columns=self.df.columns[2:])
            df = pd.concat([self.df[["ground_truth_safety_margin", "preds_safety_margin"]], df_input], axis=1)

            print(df)

            df.to_csv("./data/normalized_data.csv", index=False)

            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.pairplot(df, kind="hist", diag_kind="hist",
                         x_vars=["circle_0_channel_0_r" , "circle_0_channel_1_r", "circle_0_channel_2_r"],
                         y_vars=["circle_0_channel_0_r" , "circle_0_channel_1_r", "circle_0_channel_2_r"]
                         )
            plt.show()

    def transform_add_noise(self, tensor):
        req_shape = tensor.view(-1, 3, 3).shape
        noise = torch.randn(*req_shape) * self.preproc_error_std
        return (tensor.view(-1, 3, 3) + noise).view(*tensor.shape)


training_data = CustomDataSet(data_csv, 0, 0.7, correct_imbalance=True)
validation_data = CustomDataSet(data_csv, 0.7, 0.9)
test_data = CustomDataSet(data_csv, 0.9, 1.)

train_sampler = WeightedRandomSampler(training_data.weights, replacement=True, num_samples=len(training_data))
# validation_sampler = WeightedRandomSampler(validation_data.weights, replacement=True, num_samples=len(training_data))


batch_size = 2024
train_dataloader = DataLoader(training_data,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              # shuffle=True
                              )

validation_dataloader = DataLoader(validation_data,
                                   # sampler=validation_sampler,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=24, pin_memory=True, drop_last=False
                                   )
test_dataloader = DataLoader(test_data,
                             batch_size=batch_size, shuffle=False,
                             num_workers=24, pin_memory=True, drop_last=False
                             )


class SafetyMarginDNN(nn.Module):
    def __init__(self):
        super(SafetyMarginDNN, self).__init__()

        dropout = 0.5
        self.linear_model = nn.Sequential(
            nn.Linear(324, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        preds = self.linear_model(x)
        return preds


class LightningModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SafetyMarginDNN()
        self.loss_fn_module = nn.MSELoss()

    def loss_fn(self, preds, y):

        # preds = torch.logit(preds, 1e-6)
        # y = torch.logit(y, 1e-6)

        mse = self.loss_fn_module(preds, y)

        return torch.log(mse)

    def forward(self, x, y=None, *args):
        return self.model(x), y

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        self.log("test_loss", loss, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        preds = self.model(x)
        return preds # , y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        config_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                        mode="min",
                                                                        factor=0.5,
                                                                        patience=8),
                "monitor": "train_loss",
                "strict": True
            }
        }
        return config_dict


if __name__ == "__main__":
    model = LightningModel()

    # train model
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=128,
        log_every_n_steps=2,
        precision="32",
        default_root_dir=Path("./data")
    )
    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=validation_dataloader)

    trainer.test(model, dataloaders=test_dataloader)

    pred = trainer.predict(model, test_dataloader)
    pred = torch.concat(pred, dim=0).flatten()
    truth = test_dataloader.dataset.ground_truth.flatten()

    error = torch.sqrt(torch.square(pred - truth))

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.pairplot(pd.DataFrame({"pred": pred.flatten(),
                               "truth": truth.flatten(),
                               "error": error.flatten()}),
                 kind="hist", diag_kind="hist",
                 x_vars=["pred", "truth", "error"],
                 y_vars=["pred", "truth", "error"]
                 )
    plt.show()

    sns.lineplot(x=range(len(pred)), y=pred, label="pred")
    sns.lineplot(x=range(len(truth)), y=truth, label="truth")

    plt.show()




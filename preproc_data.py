import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import h5py

def get_data():
    dry = Path("./data/curv_dry.pkl")
    dry = pickle.load(open(dry, "rb"))
    dry = np.array(dry, dtype=float)
    shape = dry.shape
    # image/row, channel, circle, (x, y, r)
    dry = dry.reshape(shape[0], 3, -1, shape[-1])
    image_size = np.array([224, 224])
    print(dry[0, 0, 0])
    dry[:, :, :, :2] -= (image_size / 2)
    # image/row, circle, channel, (x, y, r)
    dry = np.moveaxis(dry, 1, 2)
    sm = Path("./data/test_dataset_gamma.csv")
    sm = pd.read_csv(sm, names=["ground_truth_safety_margin"])

    preds = Path("./data/curv_dry_prediction.pkl")
    preds = np.array(pickle.load(open(preds, "rb")))
    df_preds = pd.DataFrame(preds, columns=["preds_safety_margin"])

    with h5py.File('./data/training_data.h5', 'w') as f:
        group = f.create_group('curv_dry')
        group.create_dataset('input_data', data=dry, dtype='float64', compression="gzip", compression_opts=9)
        group.create_dataset('ground_truth_safety_margin', data=sm.to_numpy().flatten(), dtype='float64', compression="gzip", compression_opts=9)
        group.create_dataset('preds_safety_margin', data=preds, dtype='float64', compression="gzip", compression_opts=9)

    col_names = ["point", "circle", "channel"]
    data_axes = ["x", "y", "r"]

    dry_col_names = []
    for i in range(dry.shape[1]):  # circle
        for j in range(dry.shape[2]):  # channel
            for k in range(dry.shape[3]):  # data_axes
                dry_col_names.append(
                    f"circle_{i}_channel_{j}_{data_axes[k]}"
                )

    df_data = pd.DataFrame(dry.reshape(dry.shape[0], -1), columns=dry_col_names)
    df = pd.concat([sm, df_preds], axis=1)
    df = pd.concat([df, df_data], axis=1)

    df.to_csv("./data/full_data.csv", index=False)

    return df, dry

if __name__ == "__main__":
    # df, dry = get_data()

    df = pd.read_csv(Path("./data/full_data.csv"))
    df["error"] = df["ground_truth_safety_margin"] - df["preds_safety_margin"]
    sns.pairplot(df, kind="hist", diag_kind="hist",
                 x_vars=["ground_truth_safety_margin", "preds_safety_margin", "error"],
                 y_vars=["ground_truth_safety_margin", "preds_safety_margin", "error"]
                 )
    plt.show()
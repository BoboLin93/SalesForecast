import os
import torch
import pandas as pd
import numpy
from torch.serialization import safe_globals
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting import TemporalFusionTransformer

def load_dataset(dataset_path: str) -> TimeSeriesDataSet:
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"❌ 找不到 dataset.pkl，請確認路徑：{dataset_path}")

    # 先擴充允許的類別，再用 safe_globals 包起來安全載入
    torch.serialization.add_safe_globals([
        TimeSeriesDataSet, 
        numpy.dtype, 
        numpy.core.multiarray.scalar, 
        numpy.int64
    ])

    with safe_globals([TimeSeriesDataSet, numpy.dtype, numpy.core.multiarray.scalar, numpy.int64]):
        dataset = torch.load(dataset_path, weights_only=False)

    if not isinstance(dataset, TimeSeriesDataSet):
        raise TypeError(f"載入物件非 TimeSeriesDataSet，請確認檔案正確，實際型別：{type(dataset)}")
    return dataset


def load_model(ckpt_path: str) -> TemporalFusionTransformer:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"❌ 找不到模型檔案，請確認路徑：{ckpt_path}")
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    return model


def load_raw_data(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"❌ 找不到 CSV 檔案，請確認路徑：{csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["client", "opportunity", "datetime"])
    df["time_idx"] = df.groupby("client").cumcount()
    return df


# 範例 main 使用
if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\Bobo\AIF\tft_dashboard\data\dataset.pkl"
    MODEL_CKPT_PATH = r"C:\Users\Bobo\AIF\tft_dashboard\models\tft_model.ckpt"
    CSV_DATA_PATH = r"C:\Users\Bobo\AIF\tft_dashboard\data\sample_sales_data.csv"

    try:
        dataset = load_dataset(DATASET_PATH)
        print("✅ Dataset 載入成功")
        model = load_model(MODEL_CKPT_PATH)
        print("✅ 模型載入成功")
        df = load_raw_data(CSV_DATA_PATH)
        print("✅ 原始資料載入成功")
    except Exception as e:
        print(f"❌ 載入失敗：{e}")

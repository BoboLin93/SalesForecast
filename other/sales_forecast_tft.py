# sales_forecast_tft.py

import pandas as pd
import torch
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    QuantileLoss,
)
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor


# ✅ 讀取 CSV 檔案
df = pd.read_csv(r"C:\Users\Bobo\AIF\TFT\data\sample_sales_data.csv", encoding="utf-8-sig")

# ✅ 資料前處理
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values(["client", "opportunity", "datetime"])
df["time_idx"] = df.groupby("client").cumcount()

# ✅ 設定參數
max_encoder_length = 30
max_prediction_length = 7

# ✅ 建立 TimeSeriesDataSet
dataset = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="order_amount",
    group_ids=["client", "opportunity"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["client", "opportunity"],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["order_amount"],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# ✅ 資料切分
training_cutoff = df["time_idx"].max() - max_prediction_length * 2
train_dataset = TimeSeriesDataSet.from_dataset(dataset, df[df.time_idx <= training_cutoff])
val_dataset = TimeSeriesDataSet.from_dataset(dataset, df, predict=True, stop_randomization=True)

train_dataloader = train_dataset.to_dataloader(train=True, batch_size=64)
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=64)

# ✅ 建立原始 TFT 模型
raw_tft = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=QuantileLoss(),
    log_interval=10,
    log_val_interval=1,
    reduce_on_plateau_patience=3,
)

# ✅ 包裝成 LightningModule
class TFTLightningWrapper(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = model.loss

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.03)

# ✅ 包裝並訓練模型
tft = TFTLightningWrapper(raw_tft)

trainer = Trainer(
    max_epochs=20,
    accelerator="auto",
    devices=1,
    gradient_clip_val=0.1,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ],
    log_every_n_steps=10,
)

trainer.fit(tft, train_dataloader, val_dataloader)

# ✅ 預測
pred, true = raw_tft.predict(val_dataloader, return_y=True)

# ✅ 顯示部分預測結果
print("預測結果前五筆：")
# print(pred[0][:5])  # pred 通常是 (batch_size, target_size) 型態，視情況改用 pred[:5]
# print("真實值前五筆：")
# print(true[0][:5])
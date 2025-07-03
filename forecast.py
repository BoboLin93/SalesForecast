from pytorch_forecasting.data.timeseries import TimeSeriesDataSet

def predict_for_client(model, dataset, df_client):
    val_dataset = TimeSeriesDataSet.from_dataset(dataset, df_client, predict=True, stop_randomization=True)
    val_dataloader = val_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    predictions, x = model.predict(val_dataloader, return_x=True)
    return predictions, x

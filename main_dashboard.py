from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from data_loader import load_dataset, load_model, load_raw_data
from forecast import predict_for_client
from visualization import create_forecast_figure

# 路徑設定
DATASET_PATH = r"C:\Users\Bobo\AIF\tft_dashboard\data\dataset.pkl"
MODEL_PATH = "tft_model.ckpt"
CSV_PATH = "data/sample_sales_data.csv"

# 載入資料與模型
dataset = load_dataset(DATASET_PATH)
tft_model = load_model(MODEL_PATH)
df = load_raw_data(CSV_PATH)

# 初始化 Dash
app = Dash(__name__)
app.title = "TFT Forecast Dashboard"

app.layout = html.Div([
    html.H1("📈 TFT Sales Forecast Dashboard"),
    dcc.Dropdown(
        id="client-dropdown",
        options=[{"label": c, "value": c} for c in df["client"].unique()],
        value=df["client"].unique()[0],
        style={"width": "50%"}
    ),
    dcc.Graph(id="forecast-graph")
])

@app.callback(
    Output("forecast-graph", "figure"),
    Input("client-dropdown", "value")
)
def update_graph(selected_client):
    df_client = df[df["client"] == selected_client]
    predictions, x = predict_for_client(tft_model, dataset, df_client)

    decoder_time_idx = x["decoder_time_idx"][0].detach().cpu().numpy()
    decoder_dates = df_client.reset_index(drop=True).iloc[decoder_time_idx]["datetime"].values

    median_pred = predictions[0][..., 0].detach().cpu().numpy()
    q10 = predictions[0][..., 1].detach().cpu().numpy()
    q90 = predictions[0][..., 2].detach().cpu().numpy()
    true_vals = x["decoder_target"][0].detach().cpu().numpy()

    return create_forecast_figure(decoder_dates, true_vals, median_pred, q10, q90, selected_client)

if __name__ == "__main__":
    app.run_server(debug=True)

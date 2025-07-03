import plotly.graph_objs as go

def create_forecast_figure(decoder_dates, true_vals, median_pred, q10, q90, client_name):
    trace_actual = go.Scatter(x=decoder_dates, y=true_vals, mode='lines+markers', name='Actual', line=dict(color='black'))
    trace_predicted = go.Scatter(x=decoder_dates, y=median_pred, mode='lines+markers', name='Predicted (Median)', line=dict(color='blue'))
    trace_q10 = go.Scatter(x=decoder_dates, y=q10, mode='lines', name='10th Percentile', line=dict(dash='dot', color='lightblue'))
    trace_q90 = go.Scatter(x=decoder_dates, y=q90, mode='lines', name='90th Percentile', line=dict(dash='dot', color='lightblue'), fill='tonexty', fillcolor='rgba(173,216,230,0.2)')

    layout = go.Layout(
        title=f"📊 Sales Forecast for Client {client_name}",
        xaxis_title="Date", yaxis_title="Order Amount",
        hovermode="x unified"
    )

    return go.Figure(data=[trace_q10, trace_q90, trace_predicted, trace_actual], layout=layout)

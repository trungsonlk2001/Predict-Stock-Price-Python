from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
app = Dash(__name__)

df = pd.read_csv('data_test_SSI.csv')

fig = go.Figure([go.Scatter(x=df['date'], y=df['close'])])

app.layout = html.Div([
    html.H4('Stock price analysis'),
    dcc.Graph(figure=fig)
])


if __name__ == "__main__":
    app.run_server(debug=True)

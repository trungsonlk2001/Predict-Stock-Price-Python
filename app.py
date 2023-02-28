import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import plotly.express as px

app = dash.Dash(name='DỰ ĐOÁN CỔ PHIẾU')
server = app.server

df = pd.read_csv('data_test_VIC.csv')


# Load du lieu tu 2021 - 2022
dataset_test = pd.read_csv('data_test_VIC.csv')
real_stock_price = dataset_test.iloc[:, 4:5].values

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(df.iloc[:, 1:2].values)
# Tien hanh du doan

# regressor_VIC = load_model("model_VIC.h5")
# filename = '/content/drive/MyDrive/Năm 4/CNM/predict_stock_SSI.sav'
# pickle.dump(model, open(filename, 'wb'))
# load the model from disk
# regressor_VIC = pickle.load(open(app\predict_stock_SSI.sav, 'rb'))
dataset_test_SSI = pd.read_csv('data_test_SSI.csv')
x_test_SSI = dataset_test_SSI.drop(columns = ['close','date']).values
y_test_SSI = dataset_test_SSI[['close']].values
model_SSI = pickle.load(open('linear_predict_SSI.sav', 'rb'))
predicted_stock_price_SSI = model_SSI.predict(x_test_SSI)

dataset_test_VIC = pd.read_csv('data_test_VIC.csv')
x_test_VIC = dataset_test_VIC.drop(columns = ['close','date']).values
y_test_VIC = dataset_test_VIC[['close']].values
model_VIC = pickle.load(open('linear_predict_VIC.sav', 'rb'))
predicted_stock_price_VIC = model_VIC.predict(x_test_VIC)

# dataset_total = pd.concat((df['close'], dataset_test['close']), axis = 0)
# inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# inputs = inputs.reshape(-1,1)
# inputs = sc.transform(inputs)

# X_test = []
# no_of_sample = len(inputs)

# for i in range(60, no_of_sample):
#     X_test.append(inputs[i-60:i, 0])

# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# predicted_stock_price = regressor_VIC.predict(X_test)
# predicted_stock_price = sc.inverse_transform(predicted_stock_price)


SSI  = pd.DataFrame()
SSI['NGÀY'] = dataset_test_SSI['date'].values
SSI['GIÁ THẬT'] = dataset_test_SSI['close'].values
SSI['GIÁ DỰ ĐOÁN'] = predicted_stock_price_SSI

VIC  = pd.DataFrame()
VIC['NGÀY'] = dataset_test_VIC['date'].values
VIC['GIÁ THẬT'] = dataset_test_VIC['close'].values
VIC['GIÁ DỰ ĐOÁN'] = predicted_stock_price_VIC

fig = px.line(VIC, x='NGÀY', y=VIC.columns,
              hover_data={'NGÀY': "|%B %d, %Y"})

fig.update_layout(
    yaxis_title="GIÁ"
    
)
fig_1 = px.line(SSI, x='NGÀY', y=SSI.columns,
              hover_data={'NGÀY': "|%B %d, %Y"})

fig_1.update_layout(
    yaxis_title="GIÁ"
    
)

app.layout = html.Div([
   
    html.H1("DỰ ĐOÁN GIÁ CỔ PHIẾU VIC VÀ SSI", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       dcc.Tab(label='VIC',children=[
            html.Div([
                html.H2("Dự đoán xu hướng của giá cổ phiếu so với giá thật sử dụng mạng neural hồi quy ",style={"textAlign": "center"}),
                dcc.Graph(
                    figure=fig
                ),

                html.H2("Dự đoán xu hướng của giá cổ phiếu hien tai",style={"textAlign": "center"}),
                dcc.Graph(
                    figure=fig
                ),

                html.H2("Dự đoán xu hướng của giá cổ phiếu so với giá thật sử dụng hồi quy tuyến tính",style={"textAlign": "center"}),
                dcc.Graph(
                    figure=fig                  
                )
            ])        

        ]),
        dcc.Tab(label='SSI',children=[
            html.Div([
                html.H2("Dự đoán xu hướng của giá cổ phiếu so với giá thật sử dụng mạng neural hồi quy",style={"textAlign": "center"}),
                dcc.Graph(
                    figure=fig_1
                ),
                html.H2("Dự đoán xu hướng của giá cổ phiếu so với giá thật sử dụng hồi quy tuyến tính",style={"textAlign": "center"}),
                dcc.Graph(
                    figure=fig_1
                )
            ])        
        ])        
        
    ])
])

if __name__=='__main__':
    app.run_server(host = '0.0.0.0',port = '9898')
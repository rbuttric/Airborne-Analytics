import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle  # Import pickle for loading the pre-trained model

# Load the pre-trained model
with open('clf_model', 'rb') as model_file:
    lgbm_model = pickle.load(model_file)

flight_df = pd.read_csv('flight info.csv')

fig = go.Figure()

# Add airport markers
fig.add_trace(go.Scattergeo(
    locationmode='USA-states',
    lon=flight_df['Origin_Longitude'],
    lat=flight_df['Origin_Latitude'],
    hoverinfo='text',
    text=flight_df['Origin'],
    mode='markers',
    marker=dict(
        size=4,
        color='rgb(211, 211, 211)',
        line=dict(
            width=10,
            color='rgba(68, 68, 68, 0)'
        )
    )))

# Add flight paths
lons = np.empty(3 * len(flight_df))
lons[::3] = flight_df['Origin_Longitude']
lons[1::3] = flight_df['Dest_Longitude']
lons[2::3] = None
lats = np.empty(3 * len(flight_df))
lats[::3] = flight_df['Origin_Latitude']
lats[1::3] = flight_df['Dest_Latitude']
lats[2::3] = None

fig.add_trace(
    go.Scattergeo(
        locationmode='USA-states',
        lon=lons,
        lat=lats,
        mode='lines',
        line=dict(width=3, color='rgba(250, 50, 50, 1)'),
        opacity=0.5
    )
)

# Update layout
fig.update_layout(
    title_text='Flight Map',
    showlegend=False,
    geo=dict(
        scope='north america',
        projection_type='azimuthal equal area',
        showland=True,
        landcolor='rgb(243, 243, 243)',
        countrycolor='rgb(204, 204, 204)',
    ),
    height=700,
)

# Initialize Dash app
app = dash.Dash(__name__)

# Define unique values for the drop-down menu
route_options = [{'label': route, 'value': route} for route in flight_df['route'].unique()]

# Define app layout
app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='route-dropdown',
            options=route_options,
            value=flight_df['route'].iloc[0],
            style={'width': '50%'}
        )
    ], style={'margin-bottom': '20px'}),

    # Add seven input components
    html.Div([
        dcc.Input(id='wheels-off-time', type='number', placeholder='WheelsOff_time'),
        dcc.Input(id='wheels-on-time', type='number', placeholder='WheelsOn_time'),
        dcc.Input(id='dep-delay', type='number', placeholder='DepDelay'),
        dcc.Input(id='month', type='number', placeholder='Month'),
        dcc.Input(id='taxi-out', type='number', placeholder='TaxiOut'),
        dcc.Input(id='taxi-in', type='number', placeholder='TaxiIn'),
        dcc.Input(id='distance', type='number', placeholder='Distance'),
    ], style={'margin-bottom': '20px'}),

    # Add predict button
    html.Button('Predict', id='predict-button', n_clicks=0),

    dcc.Graph(id='flight-map', figure=fig),

    # Display the prediction result
    html.Div(id='prediction-output')
])


# Define callback to update the figure based on the selected route and input values
@app.callback(
    Output('flight-map', 'figure'),
    [
        Input('route-dropdown', 'value'),
        Input('wheels-off-time', 'value'),
        Input('wheels-on-time', 'value'),
        Input('dep-delay', 'value'),
        Input('month', 'value'),
        Input('taxi-out', 'value'),
        Input('taxi-in', 'value'),
        Input('distance', 'value'),
        Input('predict-button', 'n_clicks'),
    ]
)
def update_map(selected_route, wheels_off_time, wheels_on_time, dep_delay, month, taxi_out, taxi_in, distance,
               n_clicks):
    # Filter flight paths based on the selected route
    filtered_paths = flight_df[flight_df['route'] == selected_route]

    # Update the figure based on the input values
    updated_fig = go.Figure(fig)
    updated_fig.data[1]['lon'] = np.concatenate(
        [filtered_paths['Origin_Longitude'], filtered_paths['Dest_Longitude'], [None]])
    updated_fig.data[1]['lat'] = np.concatenate(
        [filtered_paths['Origin_Latitude'], filtered_paths['Dest_Latitude'], [None]])

    # You can use the input values (e.g., wheels_off_time, wheels_on_time, etc.) as needed
    # Update the figure or perform any other necessary actions here

    return updated_fig


# Define callback for prediction
@app.callback(
    Output('prediction-output', 'children'),
    [
        Input('predict-button', 'n_clicks'),
        Input('route-dropdown', 'value'),
        Input('wheels-off-time', 'value'),
        Input('wheels-on-time', 'value'),
        Input('dep-delay', 'value'),
        Input('month', 'value'),
        Input('taxi-out', 'value'),
        Input('taxi-in', 'value'),
        Input('distance', 'value'),
    ]
)
def predict(n_clicks, selected_route, wheels_off_time, wheels_on_time, dep_delay, month, taxi_out, taxi_in, distance):
    if n_clicks > 0:
        # Prepare input for prediction
        input_data = [[wheels_off_time, wheels_on_time, dep_delay, month, taxi_out, taxi_in, distance]]

        # Use the pre-trained model to make a prediction
        prediction = lgbm_model.predict(input_data)[0]

        return f'Predicted Value: {prediction}'

    return ''


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle
from datetime import datetime as dt
from sqlalchemy import create_engine, MetaData, Table, select
import xgboost as xgb
from datetime import datetime, timedelta
import plotly.express as px


# Load supervised models
with open('clf_model', 'rb') as model_file:
    lgbm_model = pickle.load(model_file)

with open('pt_model', 'rb') as model_file1:
    inv_model = pickle.load(model_file1)

# Load flight data
flight_df = pd.read_csv('flight info.csv')

# Create flight map
fig = go.Figure()

# Airport markers
fig.add_trace(go.Scattergeo(
    locationmode='USA-states',
    lon=flight_df['Origin_Longitude'],
    lat=flight_df['Origin_Latitude'],
    hoverinfo='text',
    text=flight_df['Origin'],
    mode='markers',
    marker=dict(
        size=3,
        color='rgb(250, 150, 150)'
    )))

# Flight paths
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
        line=dict(width=3, color='rgba(250, 0, 0, 1)'),
        opacity=0.9
    )
)

# Update layout
fig.update_layout(
    showlegend=False,
    margin=dict(l=40, r=40, b=0, t=0),
    geo=dict(
        scope='north america',
        projection_type='azimuthal equal area',
        showland=True,
        landcolor='rgb(250, 250, 250)',
        bgcolor='rgb(230, 230, 250)',
        lakecolor='rgb(230, 230, 250)'
    ),
    height=700,
)

# Define the SQL Server connection parameters
server = 'airborneanalytics.cpdhuvhrm3nv.us-east-1.rds.amazonaws.com'
database = 'airborneanalytics'
username = 'Team25'
password = 'AirborneAnalytics25'

connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"

engine = create_engine(connection_string)


def get_airport_traffic_data(user_origin):
    # Define the view and table names
    view_name = 'Airport_Traffic_With_Holiday'
    holidays_table_name = 'US_Holidays'

    # Create a connection
    connection = engine.connect()

    # Reflect the view and table structures
    metadata = MetaData()
    view = Table(view_name, metadata, autoload=True, autoload_with=engine)
    holidays_table = Table(holidays_table_name, metadata, autoload=True, autoload_with=engine)

    # Build a select query to return all columns from the view
    select_query = select([view])

    # Check if user_origin is 'ALL', if not, add the 'Airport' filter
    if user_origin != 'ALL':
        select_query = select_query.where(view.c.Airport == user_origin)

    # Execute the query and fetch all rows into a Pandas DataFrame
    daily_ap_cnt_df = pd.read_sql_query(select_query, connection)

    # Build a select query for the 'US_Holidays' table
    holidays_query = select([holidays_table])

    # Execute the query and fetch all rows into a Pandas DataFrame
    holidays_df = pd.read_sql_query(holidays_query, connection)
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])

    # Close the connection
    connection.close()

    return daily_ap_cnt_df, holidays_df


column_list = ['FlightDate', 'holiday_bool', 'Month_1', 'Month_2',
               'Month_3', 'Month_4', 'Month_5', 'Month_6',
               'Month_7', 'Month_8', 'Month_9', 'Month_10',
               'Month_11', 'Month_12', 'Holiday_4th of July', 'Holiday_Christmas Day',
               'Holiday_Christmas Eve', 'Holiday_Columbus Day', 'Holiday_Eastern Easter',
               'Holiday_Juneteenth', 'Holiday_Labor Day', 'Holiday_Labor Day Weekend',
               'Holiday_Martin Luther King, Jr. Day', 'Holiday_Memorial Day', "Holiday_New Year's Day",
               "Holiday_New Year's Eve", 'Holiday_Thanksgiving Day', 'Holiday_Thanksgiving Eve',
               "Holiday_Valentine's Day", 'Holiday_Veterans Day', "Holiday_Washington's Birthday",
               'Holiday_Western Easter', 'DayOfWeek_0', 'DayOfWeek_1', 'DayOfWeek_2',
               'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'weekofyear_1', 'weekofyear_2',
               'weekofyear_3', 'weekofyear_4', 'weekofyear_5', 'weekofyear_6',
               'weekofyear_7', 'weekofyear_8', 'weekofyear_9', 'weekofyear_10', 'weekofyear_11', 'weekofyear_12',
               'weekofyear_13', 'weekofyear_14', 'weekofyear_15', 'weekofyear_16', 'weekofyear_17', 'weekofyear_18',
               'weekofyear_19', 'weekofyear_20', 'weekofyear_21', 'weekofyear_22', 'weekofyear_23', 'weekofyear_24',
               'weekofyear_25', 'weekofyear_26', 'weekofyear_27', 'weekofyear_28', 'weekofyear_29', 'weekofyear_30',
               'weekofyear_31', 'weekofyear_32', 'weekofyear_33', 'weekofyear_34', 'weekofyear_35', 'weekofyear_36',
               'weekofyear_37', 'weekofyear_38', 'weekofyear_39', 'weekofyear_40', 'weekofyear_41', 'weekofyear_42',
               'weekofyear_43', 'weekofyear_44', 'weekofyear_45', 'weekofyear_46', 'weekofyear_47', 'weekofyear_48',
               'weekofyear_49', 'weekofyear_50', 'weekofyear_51', 'weekofyear_52']


def feature_engineering(df, with_dummies=True):
    # Ensure that the 'FlightDate' column is in datetime format
    df['FlightDate'] = pd.to_datetime(df['FlightDate'])

    # Drop the 'Date' column
    # df = df.drop(columns=['Date'])

    # Set 'FlightDate' as the index and sort the DataFrame
    df = df.set_index('FlightDate').sort_values(by='FlightDate', ascending=True)

    # Add features
    df['Month'] = df.index.month
    # df['DayofMonth'] = df.index.day
    df['DayOfWeek'] = df.index.dayofweek
    df['weekofyear'] = df.index.isocalendar().week
    df['holiday_bool'] = pd.notnull(df['Holiday']).astype(int)

    if with_dummies:
        # Perform one-hot encoding using get_dummies
        df = pd.get_dummies(df, columns=['Month', 'Holiday', 'DayOfWeek', 'weekofyear'],
                            prefix=['Month', 'Holiday', 'DayOfWeek', 'weekofyear'], dtype=int).drop(
            columns=['Airport'])
        return df
    else:
        return df


def calc_historical_stats(daily_ap_cnt_df_feats):
    # Create a dictionary to map numerical index to the corresponding month
    month_mapping = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July',
                     8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                   'November', 'December']
    # Create a dictionary to map numerical index to the corresponding day
    day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday',
                   6: 'Sunday'}
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    avg_stats = daily_ap_cnt_df_feats[['Total_AirTraffic_count', 'Month', 'DayOfWeek']].reset_index()
    avg_stats_month = avg_stats[['Total_AirTraffic_count', 'Month']].groupby(
        'Month').mean()  # .sort_values(by='Total_AirTraffic_count', ascending=False)

    # Map the 'Month' column using the created dictionary
    avg_stats_month['Month_name'] = avg_stats_month.index.map(month_mapping)

    # Convert the 'Month' column to a Categorical data type with the desired order
    avg_stats_month['Month_name'] = pd.Categorical(avg_stats_month['Month_name'], categories=month_order,
                                                   ordered=True)

    # Sort the DataFrame based on the categorical order
    avg_stats_month.sort_values('Month', inplace=True)

    avg_stats_DayOfWeek = avg_stats[['Total_AirTraffic_count', 'DayOfWeek']].groupby(
        'DayOfWeek').mean()  # .sort_values(by='Total_AirTraffic_count', ascending=False)
    # Map the 'DayOfWeek' column using the created dictionary
    avg_stats_DayOfWeek['Day'] = avg_stats_DayOfWeek.index.map(day_mapping)
    # Convert the 'Day' column to a Categorical data type with the desired order
    avg_stats_DayOfWeek['Day'] = pd.Categorical(avg_stats_DayOfWeek['Day'], categories=day_order, ordered=True)

    # Sort the DataFrame based on the categorical order
    avg_stats_DayOfWeek.sort_values('Day', inplace=True)

    return avg_stats_DayOfWeek, avg_stats_month


def user_prediction(user_date, user_origin):
    # Get historical dataset of airport traffic for the user provided airport
    daily_ap_cnt_df, holidays_df = get_airport_traffic_data(user_origin)
    daily_ap_cnt_df_feats = feature_engineering(daily_ap_cnt_df)
    daily_ap_cnt_df_feats_no_dum = feature_engineering(daily_ap_cnt_df, with_dummies=False)
    avg_stats_DayOfWeek, avg_stats_month = calc_historical_stats(daily_ap_cnt_df_feats_no_dum)

    X_train = daily_ap_cnt_df_feats[column_list[1:]]
    y_train = daily_ap_cnt_df_feats['Total_AirTraffic_count']

    # Fit model based on parameters determined from initial analysis
    xgb_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                                 n_estimators=1500,
                                 # early_stopping_rounds=50,
                                 objective='reg:linear',
                                 max_depth=2,
                                 learning_rate=0.03,
                                 n_jobs=-1, tree_method='hist')
    xgb_model.fit(X_train, y_train, verbose=100)

    start_date = datetime.strptime('01-01-2018', '%m-%d-%Y')
    end_date = datetime.strptime('12-31-2019', '%m-%d-%Y')

    # Create date range to use in prediction
    user_date = datetime.strptime(user_date, '%m-%d-%Y')
    date_list = [user_date - timedelta(days=i) for i in range(7, -8, -1)]

    # Format the dates as strings and filter based on start_date and end_date
    formatted_date_list = [date.strftime('%m-%d-%Y') for date in date_list if date < start_date or date > end_date]

    # User Query
    user_date_df = pd.DataFrame(columns=column_list)

    if len(formatted_date_list) == 0:
        # If there are no dates in the range, return the specified columns from daily_ap_cnt_df_feats
        final_df = daily_ap_cnt_df_feats[daily_ap_cnt_df_feats.index.isin(date_list)]
        final_df = final_df.rename(columns={'Total_AirTraffic_count': 'Demand'})
        return final_df['Demand'], avg_stats_DayOfWeek, avg_stats_month

    # Iterate over formatted_date_list to create rows in the DataFrame
    for date_str in formatted_date_list:
        row = [date_str] + [0] * (len(column_list) - 1)
        user_date_df = user_date_df.append(pd.Series(row, index=column_list), ignore_index=True)

    # Convert 'FlightDate' to datetime
    user_date_df['FlightDate'] = pd.to_datetime(user_date_df['FlightDate'])

    # Merge holiday dataframe
    user_date_df = user_date_df.merge(holidays_df[['Date', 'Holiday']], how='left', left_on='FlightDate',
                                      right_on='Date')
    user_date_df['holiday_bool'] = pd.notnull(user_date_df['Holiday']).astype(int)

    user_date_df = user_date_df.set_index('FlightDate')
    user_date_df['DayOfWeek'] = user_date_df.index.dayofweek
    user_date_df['Month'] = user_date_df.index.month

    # Iterate over all rows in the DataFrame
    for index, row in user_date_df.iterrows():
        # Check if 'holiday_bool' is 1
        if row['holiday_bool'] == 1:
            # Find the corresponding column name in 'Holiday' column
            holiday_name = 'Holiday_' + row['Holiday']

            # Check if the column exists in the DataFrame
            if holiday_name in user_date_df.columns:
                # Update the corresponding column value to 1
                user_date_df.loc[index, holiday_name] = 1

        # DoM = 'DayofMonth_' + str(row['DayofMonth'])
        DoW = 'DayOfWeek_' + str(row['DayOfWeek'])
        mon = 'Month_' + str(row['Month'])

        # Loop through DataFrame columns and update values
        for col in user_date_df.columns:
            # if 'DayofMonth_' in col and col == DoM:
            #     user_date_df.loc[index, col] = 1
            if 'DayOfWeek_' in col and col == DoW:
                user_date_df.loc[index, col] = 1
            elif 'Month_' in col and col == mon:
                user_date_df.loc[index, col] = 1

    # Drop unnecessary columns
    user_date_df = user_date_df.drop(columns=['Date', 'Holiday', 'DayOfWeek', 'Month']).astype(int)
    user_pred = xgb_model.predict(user_date_df[column_list[1:]])

    # Add predictions to the DataFrame
    user_date_df['Predicted_Demand'] = user_pred

    # Check if the index of user_date_df is in the index of daily_ap_cnt_df_feats
    if user_date_df.index.isin(daily_ap_cnt_df_feats.index).all():
        # If so, update the 'Prediction' column in daily_ap_cnt_df_feats
        daily_ap_cnt_df_feats['Predicted_Demand'].loc[user_date_df.index] = user_date_df['Predicted_Demand']
    else:
        # If not, add new rows to daily_ap_cnt_df_feats
        daily_ap_cnt_df_feats = pd.concat([daily_ap_cnt_df_feats, user_date_df[['Predicted_Demand']]])

    final_df = daily_ap_cnt_df_feats[daily_ap_cnt_df_feats.index.isin(date_list)]
    final_df = final_df.rename(columns={'Total_AirTraffic_count': 'Actual_Demand'})
    final_df['Demand'] = final_df['Actual_Demand'].where(final_df['Actual_Demand'].notna(),
                                                         final_df['Predicted_Demand'])
    final_df['Demand'] = final_df['Demand'].round(decimals=0)

    return final_df[['Demand']], avg_stats_DayOfWeek, avg_stats_month

#style={'backgroundColor': '#ADD8E6'},children=
# App creation
app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.H2('US Flight Delays and Forecasting'),
        dcc.Graph(id='flight-map', figure=fig),
        dcc.Graph(id='predicted-day-graph')
    ], style={'width': '49%', 'display': 'inline-block'}),

    html.Div(children=[
        html.H3('Predicting Flight Delays'),
        html.H4('Enter desired values into the boxes below, then click predict to see an approximated delay for your flight.'),

        html.Label('Origin'),
        dcc.Input(id='origin', type='text', value='DTW'),

        html.Label('Destination'),
        dcc.Input(id='destination', type='text', value='DEN'),

        html.Br(),
        html.Label('WheelsOff_time'),
        dcc.Input(id='wheels-off-time', type='number', value=900),

        html.Label('WheelsOn_time'),
        dcc.Input(id='wheels-on-time', type='number', value=1000),

        html.Br(),
        html.Label('DepDelay'),
        dcc.Input(id='dep-delay', type='number', value=0),

        html.Label('Month'),
        dcc.Input(id='month', type='number', value=10),

        html.Br(),
        html.Label('TaxiOut'),
        dcc.Input(id='taxi-out', type='number', value=10),

        html.Label('TaxiIn'),
        dcc.Input(id='taxi-in', type='number', value=5),

        html.Br(),
        html.Label('Distance'),
        dcc.Input(id='distance', type='number', value=1200),

        html.Button('Predict', id='predict-button', n_clicks=0),
        html.Div(id='prediction-output', style={'color': 'red'}),

        html.Br(),
        html.H3('Demand Forecasting'),
        html.H4('Enter your desired flight date and alter the origin and destination above to see demand statistics.'),

        html.Label('Date'),
        dcc.Input(id='date-picker',value='11-26-2020'),
        dcc.Graph(id='average-month-graph'),
        dcc.Graph(id='average-dayofweek-graph')
    ], style={'width': '50%', 'vertical-align': 'top', 'display': 'inline-block'}),
])


# Update map
@app.callback(
    Output('flight-map', 'figure'),
    [
        Input('origin', 'value'),
        Input('destination', 'value')
    ]
)
def update_map(origin, destination):
    # Filter flight paths based on the selected origin and destination
    filtered_paths = flight_df[(flight_df['Origin'] == origin) & (flight_df['Dest'] == destination)]

    # Update the figure based on the input values
    updated_fig = go.Figure(fig)
    updated_fig.data[1]['lon'] = np.concatenate(
        [filtered_paths['Origin_Longitude'], filtered_paths['Dest_Longitude'], [None]])
    updated_fig.data[1]['lat'] = np.concatenate(
        [filtered_paths['Origin_Latitude'], filtered_paths['Dest_Latitude'], [None]])

    return updated_fig


# Callback for predictions
@app.callback(
    Output('prediction-output', 'children'),
    [
        Input('predict-button', 'n_clicks'),
        Input('origin', 'value'),
        Input('destination', 'value'),
        Input('wheels-off-time', 'value'),
        Input('wheels-on-time', 'value'),
        Input('dep-delay', 'value'),
        Input('month', 'value'),
        Input('taxi-out', 'value'),
        Input('taxi-in', 'value'),
        Input('distance', 'value'),
    ]
)
def predict(n_clicks, origin, destination, wheels_off_time, wheels_on_time, dep_delay, month, taxi_out, taxi_in,
            distance):
    # 1. concatenate origin and dest
    route = origin.upper() + '-' + destination.upper()

    # 2. calculate the time between wheels off and wheels on
    # Add leading 0s if it is less than 4 digits
    wheels_off_time = f'{wheels_off_time // 100:02d}:{wheels_off_time % 100:02d}'
    wheels_on_time = f'{wheels_on_time // 100:02d}:{wheels_on_time % 100:02d}'

    # Convert the three or four digits to 'hh:mm' format
    wheels_off_time = pd.to_datetime(wheels_off_time, format='%H:%M')
    wheels_on_time = pd.to_datetime(wheels_on_time, format='%H:%M')

    # Calculate the time difference in minutes
    TimeDifferenceSeconds_wheels = (wheels_on_time - wheels_off_time).seconds
    TimeDifferenceMinutes_wheels = TimeDifferenceSeconds_wheels / 60
    TimeDifferenceMinutes_wheels = TimeDifferenceMinutes_wheels + 24 * 60 if TimeDifferenceMinutes_wheels < 0 else TimeDifferenceMinutes_wheels

    # 3. put all inputs into a dataframe
    data = {
        'DepDelay': dep_delay,
        'Month': month,
        'route': route,
        'TaxiOut': taxi_out,
        'TaxiIn': taxi_in,
        'TimeDifferenceMinutes_wheels': TimeDifferenceMinutes_wheels,
        'Distance': distance
    }

    df = pd.DataFrame(data, index=[0])
    df.loc[:, 'route'] = df['route'].astype('category')

    # 4. Predict
    pred = lgbm_model.predict(df, num_iteration=lgbm_model.best_iteration)

    # 5. Inverse the prediction result
    inversed_pred = round(inv_model.inverse_transform(pred.reshape(-1, 1))[0][0] * -1,2)

    return f'Predicted Delay: {inversed_pred} minute delay'

    # Define callback to update graphs based on user input
    @app.callback(
        [Output('predicted-day-graph', 'figure'),
         Output('average-month-graph', 'figure'),
         Output('average-dayofweek-graph', 'figure')],
        [Input('date-picker', 'value'),
         Input('Origin', 'value')]
    )
    def update_graphs(selected_date, selected_origin):
        user_date_dt = pd.to_datetime(selected_date)
        user_month = user_date_dt.strftime('%B')
        user_day = user_date_dt.strftime('%A')

        pred_df, avg_stats_DayOfWeek, avg_stats_month = user_prediction(selected_date, selected_origin)

        # Predicted Day Graph
        fig_pred_day = px.line(pred_df, x=pred_df.index, y='Demand',
                               title=f"Predicted Inbound/Outbound Flights at {selected_origin} on {selected_date}")

        fig_pred_day.add_shape(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=user_date_dt,
                y0=0,
                x1=user_date_dt,
                y1=1,
                fillcolor="red",
                opacity=0.2,
                layer="below",
                line=dict(color="red", width=60)
            )
        )
        fig_pred_day.update_layout(title_x=0.5)
        fig_pred_day.update_layout(height=300, width=980)
        fig_pred_day.update_yaxes(title_text="Total Traffic")

        # Average Month Graph
        fig_month = px.line(avg_stats_month, x='Month_name', y='Total_AirTraffic_count',
                            title=f"Average Traffic at {selected_origin} by Month")

        fig_month.add_shape(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=user_month,
                y0=0,
                x1=user_month,
                y1=1,
                fillcolor="red",
                opacity=0.2,
                layer="below",
                line=dict(color="red", width=60)
            )
        )
        fig_month.update_layout(title_x=0.5)
        fig_month.update_layout(height=300, width=980)
        fig_month.update_yaxes(title_text="Total Traffic")
        fig_month.update_xaxes(title_text="Month")

        # Average Day of Week Graph
        fig_day = px.line(avg_stats_DayOfWeek, x='Day', y='Total_AirTraffic_count',
                          title=f"Average Traffic at {selected_origin} by Day of Week")

        fig_day.add_shape(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=user_day,
                y0=0,
                x1=user_day,
                y1=1,
                fillcolor="red",
                opacity=0.2,
                layer="below",
                line=dict(color="red", width=60)
            )
        )
        fig_day.update_layout(title_x=0.5)
        fig_day.update_layout(height=300, width=980)
        fig_day.update_yaxes(title_text="Total Traffic")

        return fig_pred_day, fig_month, fig_day


if __name__ == '__main__':
    app.run_server(debug=True)

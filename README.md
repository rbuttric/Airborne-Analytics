
![Airborne Analytics](https://github.com/rbuttric/Airborne-Analytics/assets/93159436/46cde317-3331-41a0-af77-930c33c1e8cb)

# Airline Delay Prediction and Airport Demand Forecasting
## Overview
Airline flight demand forecasting is a critical aspect of the aviation industry, influencing decisions across various operational dimensions. From strategic route expansion to optimizing staffing levels, accurate forecasting empowers airlines with insights to navigate the dynamic landscape of air travel.

This project is dedicated to predicting airline delays and forecasting airport demand, with a focus on providing valuable information for consumers booking future flights. Our model aims to empower passengers by offering relevant statistics and graphs related to their selected departure airport and date. Users can input their preferred departure date and airport, and the model will generate insights into historical and forecasted traffic patterns. This data can assist passengers in making informed decisions about arrival times, exploring alternative dates with lower demand, or anticipating potential delays due to high traffic volumes.

In addition to forecasting airline flight demand, a crucial component of our project involves predicting flight delays, a common concern for air travelers. Understanding the factors contributing to delays and providing accurate predictions is essential for passengers planning their journeys. Our model analyzes historical flight data, considering variables such as flight route, air traffic, flight distance and other relevant factors, to provide insights into the likelihood of delays. By incorporating this predictive element, we aim to enhance the overall travel experience for users, allowing them to proactively plan and adjust their itineraries based on potential delays.

## Key Features
- User-Friendly Interface: Our model provides a straightforward interface where users can input their desired departure date and airport to receive tailored insights.
- Historical and Forecasted Data: Users gain access to both historical and forecasted traffic patterns, aiding in decision-making for future flights.
- Peace of Mind for Passengers: By predicting departure delays, the model helps passengers plan their itineraries more effectively and reduces the stress associated with potential delays.
- Focused Analysis on United Airlines: The analysis centers on the domestic flight records of United Airlines for the year 2018.

## How It Works
- User Input: Users input their preferred departure date and airport.
- Data Analysis: The model analyzes historical flight data, providing insights into past traffic patterns.
- Forecasting: Using advanced forecasting techniques, the model predicts future demand and potential delays.
- Visualizations: Relevant statistics and graphs are generated, offering users a clear understanding of the patterns and trends.

## Why It Matters
- Informed Decision-Making: Passengers can make informed decisions about their travel plans based on historical and forecasted data.
- Stress Reduction: Predicting departure delays provides passengers with peace of mind and the ability to proactively adjust their itineraries.
- Strategic Insights for Airlines: Airlines can use demand forecasting to optimize operations, enhance route planning, and improve overall efficiency.

## Dataset
This project focuses on the analysis of US Airline On-Time Performance Data from Jan 2018 to Dec 2019, which contains approximately 14 million flight records across 363 airports and 18 airlines. 
https://www.transtats.bts.gov/databases.asp?Z1qr_VQ=E&Z1qr_Qr5p=N8vn6v10&f7owrp6_VQF=D

There is also a supplemental dataset that contains the dates of major US holidays from 2018 to 2024.  
https://www.kaggle.com/datasets/donnetew/us-holiday-dates-2004-2021

## Technical Requirements

- Pandas (v1.5.3): Data manipulation and analysis library; provides data structures like dataframes for efficient data handling.
- Numpy (1.24.3): Numerical computing library; supports large, multi-dimensional arrays and matrices, along with mathematical functions to operate on them.
- Matplotlib (v3.6.0), seaborn (v0.12.2), plotly (v 5.9.0): Plotting libraries; used for creating static, animated, and interactive visualizations in Python.
- Xgboost (v2.0.2): Gradient boosting library designed for speed and performance; used for supervised machine learning tasks, especially in predictive modeling and classification.
- Statsmodels (v0.14.0): Statistical modeling library; provides classes and functions for estimating and testing statistical models.
- Scipy (v1.10.1): Scientific computing library; builds on NumPy and provides additional functionality for optimization, signal processing, statistics, and more.
- Pmdarima (2.0.4): AutoARIMA (AutoRegressive Integrated Moving Average) library; used for time series analysis and forecasting.
- Scikit-learn (v1.2.2): Machine learning library; provides simple and efficient tools for data mining and data analysis, built on NumPy, SciPy, and matplotlib.
- Sqlalchemy (v1.4.39): SQL toolkit and Object-Relational Mapping (ORM) library; used for communicating with relational databases and performing database operations.
- Datetime: Module for working with dates and times; provides classes for manipulating dates and times in both simple and complex ways.

## Files and order of execution
### AWS Data Loads
- AWS SQL Server DB Load.ipynb
-- This file contains the code that looped through the on-time performace data stored locally, then inserted it into the AWS database in chunks of 10,000.
### Demand Forecasting
- Demand Forecasting - Analysis-Modeling.ipynb
-- This file contains all code related to data cleaning, feature engineering, model selection and model evaluation. Authorization is needed to connect to the AWS database to pull all on-time performace data, so
  included in the funtion get_airport_traffic_data there is an option to set sample_data=True, this will pull data from daily_ap_cnt_df.csv instead of querying the database.
- Demand Forecasting - Dashboard.ipynb
-- This file contains the best performing model found in Demand Forecasting - Analysis-Modeling.ipynb, user inputs for origin/destination airport and flight date, and outputs demand predictions and historic demand trends. This file also contains the function that allows the pulling of sample data. 

Feel free to explore our project and leverage the power of data-driven insights for a smoother and more informed air travel experience!

# Trains Analysis
# Import required libraries for data processing and analysis
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot to visualize correlation

# Activate pandas conversion for R objects
pandas2ri.activate()

# Load RData file using R
base = importr('base')
base.load("trainsData.RData")

# Extract data objects from R environment
training_data_r = robjects.globalenv['trainingData']
test_data_r = robjects.globalenv['testData']
historical_congestion_r = robjects.globalenv['historicalCongestion']

def convert_historical_congestion(historical_congestion_r):
    """
    Convert historical congestion data from R format to pandas DataFrame.
    
    Parameters:
    historical_congestion_r (R object): Historical congestion data from R
    
    Returns:
    pandas.DataFrame: Processed historical congestion data with proper types
    """
    try:
        # Transpose data to get correct shape
        data = np.array(historical_congestion_r).T
        
        # Get column names from R object
        column_names = list(historical_congestion_r.colnames)
        
        # Create pandas DataFrame with proper column names
        df = pd.DataFrame(data, columns=column_names)
        
        # Convert data types
        df['Day'] = df['Day'].astype(str)
        df['Hour'] = df['Hour'].astype(int)
        
        # Convert numerical columns to float
        numeric_columns = [col for col in df.columns if col not in ['Day', 'Hour']]
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        return df
        
    except Exception as e:
        print(f"Error converting historical congestion data: {str(e)}")
        raise
def convert_train_data(training_data_r):
    """
    Convert train data from R format to pandas DataFrame, preserving all features.
    
    Parameters:
    training_data_r (R object): Train timing data from R
    
    Returns:
    pandas.DataFrame: Complete train data with all original features
    """
    train_data_list = []
    
    for train in training_data_r:
        try:
            # Extract all components of train data
            timings_data = train[0]
            congestion_data = train[1]
            arrival_data = train[2]
            
            # Get column names for each component
            timing_cols = list(timings_data.names)
            congestion_cols = list(congestion_data.names)
            arrival_cols = list(arrival_data.names)
            
            # Create a dictionary for all timing features
            timing_dict = {}
            for i, col in enumerate(timing_cols):
                timing_dict[f"timing_{col}"] = [val for val in timings_data[i]]
            
            # Create a dictionary for all congestion features
            congestion_dict = {}
            for i, col in enumerate(congestion_cols):
                congestion_dict[f"congestion_{col}"] = congestion_data[i][0]
            
            # Create a dictionary for all arrival features (if available)
            arrival_dict = {}
            if len(arrival_data) > 0:
                for i, col in enumerate(arrival_cols):
                    arrival_dict[f"arrival_{col}"] = arrival_data[i][0]
            
            # Combine all features
            train_info = {**timing_dict, **congestion_dict, **arrival_dict}
            train_data_list.append(train_info)
            
        except Exception as e:
            print(f"Error processing train data: {str(e)}")
            continue
    
    return pd.DataFrame(train_data_list)

def process_features(df):
    """
    Process and engineer features for train delay prediction.
    
    Parameters:
    df (pandas.DataFrame): Raw train data
    
    Returns:
    pandas.DataFrame: Processed data with engineered features
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Extract fourth elements from list columns first
    list_columns = [
        'timing_departure.time',
        'timing_departure.schedule',
        'timing_arrival.time',
        'timing_arrival.schedule',
        'timing_day.week'
    ]
    
    for col in list_columns:
        df[col] = df[col].apply(lambda x: x[-1])
    
    # Convert time strings to seconds
    def time_to_seconds(time_str):
        hours, minutes, seconds = map(int, time_str.split(':'))
        return hours * 3600 + minutes * 60 + seconds
    
    # Create time-based features
    df['departure_delay'] = df.apply(lambda x: 
        time_to_seconds(x['timing_departure.time']) - time_to_seconds(x['timing_departure.schedule']), axis=1)
    
    df['arrival_delay'] = df.apply(lambda x: 
        time_to_seconds(x['timing_arrival.time']) - time_to_seconds(x['timing_arrival.schedule']), axis=1)
    
    # Extract hour information from the already processed departure time
    df['hour'] = df['timing_departure.time'].apply(lambda x: int(x.split(':')[0]))
    
    # Convert days to numerical features - assuming timing_day.week contains the day information
    df['day_of_week'] = pd.Categorical(df['timing_day.week'].apply(lambda x: x[3])).codes
    
    # Create peak hour indicator
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0)
    
    # Calculate average station delays
    df['avg_station_delay'] = (df['congestion_Leeds.av.delay'] + 
                              df['congestion_Sheffield.av.delay'] + 
                              df['congestion_Nottingham.av.delay']) / 3
    
    # Calculate total congestion
    df['total_congestion'] = (df['congestion_Leeds.trains'] + 
                             df['congestion_Sheffield.trains'] + 
                             df['congestion_Nottingham.trains'])
    
    df.plot(x='day_of_week', y='arrival_delay', kind='line')
    plt.show()

    return df
def train_delay_model(df):
    """
    Train a random forest model for predicting train delays.
    
    Parameters:
    df (pandas.DataFrame): Processed train data
    
    Returns:
    tuple: (trained model, scaler, RMSE score, feature list)
    """
    # Select features for modeling
    features = ['day_of_week', 'departure_delay', 'arrival_delay',
               'congestion_Leeds.trains', 'congestion_Leeds.av.delay',
               'congestion_Sheffield.trains', 'congestion_Sheffield.av.delay',
               'hour', 'is_peak_hour', 'avg_station_delay', 'total_congestion']
    
    X = df[features]
    y = df['arrival_delay.secs']
    
    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model using linear regression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    val_predictions = model.predict(X_val_scaled)
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

    
    return model, scaler, rmse, features
historical_df = convert_historical_congestion(historical_congestion_r)
historical_df.head(6)

df = process_features(convert_train_data(training_data_r))

plt.figure(figsize=(12, 6))
sns.lineplot(x='hour', y='arrival_delay', data=df, marker='o')
plt.xlabel("Hour of the Day")
plt.ylabel("Average Arrival Delay (seconds)")
plt.title("Train Arrival Delays Throughout the Day")
plt.grid()
plt.show()

stations = ['congestion_Leeds.av.delay', 'congestion_Sheffield.av.delay', 'congestion_Nottingham.av.delay']
df_melted = df.melt(id_vars=['hour'], value_vars=stations, var_name="Station", value_name="Average Delay")

plt.figure(figsize=(12, 6))
sns.lineplot(x='hour', y='Average Delay', hue='Station', data=df_melted, marker='o')
plt.xlabel("Hour of the Day")
plt.ylabel("Average Delay (seconds)")
plt.title("Average Station Delay by Hour")
plt.legend(title="Station")
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['total_congestion'], y=df['arrival_delay'])
plt.xlabel("Total Congestion (Number of Trains)")
plt.ylabel("Arrival Delay (seconds)")
plt.title("Effect of Congestion on Train Delays")
plt.grid()
plt.show()

#Initialise days of the week
day_mapping = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}

# Mapping to replace numbers with day names
df['day_of_week'] = df['day_of_week'].map(day_mapping)

plt.figure(figsize=(12, 6))
sns.boxplot(x='day_of_week', y='arrival_delay', data=df, order=day_mapping.values())
plt.xlabel("Day of the Week")
plt.ylabel("Arrival Delay (seconds)")
plt.title("Train Delay Distribution by Day of the Week")
plt.grid()
plt.show()

# Check non-numeric columns
non_numeric_cols = df.select_dtypes(exclude=['number']).columns
print("Non-numeric columns:", non_numeric_cols.tolist())

# Convert list columns to scalar values
for col in non_numeric_cols:
    df[col] = df[col].apply(lambda x: x[-1] if isinstance(x, list) else x)

# Convert now-cleaned columns to numeric
df[non_numeric_cols] = df[non_numeric_cols].apply(pd.to_numeric, errors='coerce')

# Drop remaining non-numeric columns
df_cleaned = df.select_dtypes(include=['number'])

plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha="right")
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Define peak hours
df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0)

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['is_peak_hour'], y=df['arrival_delay'])
plt.xticks(ticks=[0, 1], labels=["Non-Peak", "Peak"])
plt.xlabel("Time of Day")
plt.ylabel("Arrival Delay (seconds)")
plt.title("Train Delays During Peak vs. Non-Peak Hours")
plt.grid()
plt.show()

#Initialise days of the week
day_mapping = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}

# Apply mapping to replace numbers with day names
df['day_of_week'] = df['day_of_week'].map(day_mapping)

plt.figure(figsize=(12, 6))
sns.boxplot(x='day_of_week', y='arrival_delay', data=df, order=day_mapping.values())  # Ensure order is correct
plt.xlabel("Day of the Week")
plt.ylabel("Arrival Delay (seconds)")
plt.title("Train Delay Distribution by Day of the Week")
plt.grid()
plt.show()

# Check for non-numeric columns
non_numeric_cols = df.select_dtypes(exclude=['number']).columns
print("Non-numeric columns:", non_numeric_cols.tolist())

# Convert any list columns to scalar values
for col in non_numeric_cols:
    df[col] = df[col].apply(lambda x: x[-1] if isinstance(x, list) else x)

# Convert columns to numeric
df[non_numeric_cols] = df[non_numeric_cols].apply(pd.to_numeric, errors='coerce')

# Drop remaining non-numeric columns
df_cleaned = df.select_dtypes(include=['number'])

plt.figure(figsize=(12, 8)) 
heatmap = sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Rotate X and Y axis 
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha="right")
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Define peak hours
df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0)

stations = ['Leeds', 'Sheffield', 'Nottingham']

plt.figure(figsize=(12, 6))

for station in stations:
    station_col_delay = f'congestion_{station}.av.delay'  
    sns.boxplot(x=df['is_peak_hour'], y=df[station_col_delay], label=station)

plt.xticks(ticks=[0, 1], labels=["Non-Peak", "Peak"])
plt.xlabel("Time of Day")
plt.ylabel("Average Delay per Station (seconds)")
plt.title("Station-wise Delays During Peak vs. Non-Peak Hours")
plt.legend(title='Stations')
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['is_peak_hour'], y=df['arrival_delay'])
plt.xticks(ticks=[0, 1], labels=["Non-Peak", "Peak"])
plt.xlabel("Time of Day")
plt.ylabel("Overall Arrival Delay (seconds)")
plt.title("Overall Train Delays During Peak vs. Non-Peak Hours")
plt.grid()
plt.show()

print(convert_train_data)
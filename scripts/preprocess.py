from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, BooleanType
import math

def check_dataset (sdf): 
    '''Output missing values and descriptive statistics in the given dataset'''
    print("Missing values:")
    missing_values = sdf.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) 
                                        for c in sdf.columns])
    missing_values.show()

    # Check range of pickup time
    print("tpep_pickup_datetime:\n\tLatest:", sdf.agg({"tpep_pickup_datetime": "max"}).collect()[0][0],
        "\n\tEarliest:", sdf.agg({"tpep_pickup_datetime": "min"}).collect()[0][0])

     # Check range of dropoff time
    print("\ntpep_dropoff_datetime:\n\tLatest:", sdf.agg({"tpep_dropoff_datetime": "max"}).collect()[0][0],
        "\n\tEarliest:", sdf.agg({"tpep_dropoff_datetime": "min"}).collect()[0][0], '\n')
    
    # Descriptive statistics
    print("Descriptive statistics")
    description = sdf.select(["trip_distance", "pulocationid", "dolocationid"]).describe()
    description.show()

def filter_columns (set_name, sdf): 
    '''Filter features in the given dataframe to be within reasonable ranges
    and output the filtered dataframe'''

    if set_name == 'train': 
        sdf = sdf.filter(
            # Filter for training data: 1st Dec 2022 to end of 31st May 2023
            (F.col("tpep_pickup_datetime").between("2022-12-01 00:00:00", "2023-05-31 23:59:59")) & 
            (F.col("tpep_dropoff_datetime").between("2022-12-01 00:00:00", "2023-05-31 23:59:59")) 
        )    
    elif set_name == 'test':
        # Filter for testing data: 1st Jan 2024 to 31st May 2024
        sdf = sdf.filter(
            (F.col("tpep_pickup_datetime").between("2024-01-01 00:00:00", "2024-05-31 23:59:59")) &
            (F.col("tpep_dropoff_datetime").between("2024-01-01 00:00:00", "2024-03-31 23:59:59"))
        )
    else:
        raise ValueError("set_name should be either be 'train' or 'test'")

    # ensure pickup and dropoff location are well defined in taxi+_zone_lookup.csv
    sdf = sdf.filter(
        (F.col("pulocationid").between(1, 263)) & 
        (F.col("dolocationid").between(1, 263)) &

        # only interested in trips that go for at least 1 mile and at most 50 miles
        (F.col("trip_distance").between(1, 50))     
    )
    return sdf

def check_outliers(sdf, column_name):
    '''Using IQR rule for N > 100 to flag outliers in the given column and given dataframe
      Show flagged outliers and output the lower, upper bound '''
    
    N = sdf.count()
    
    # Calculate IQR
    quantiles = sdf.approxQuantile(column_name, [0.25, 0.75], 0.01)
    Q1 = quantiles[0]
    Q3 = quantiles[1]
    IQR = Q3 - Q1

    # Calculate the scaling factor
    scaling_factor = math.sqrt(math.log(N)) - 0.5

    # Calculate lower and upper bounds
    lower_bound = Q1 - (scaling_factor * IQR)
    upper_bound = Q3 + (scaling_factor * IQR)

    # Filter rows that are considered outliers
    outliers_df = sdf.filter((F.col(column_name) < lower_bound) | (F.col(column_name) > upper_bound))

    # Show outliers
    outliers_df.orderBy(F.col(column_name).desc()).show()

    return lower_bound, upper_bound

def find_low_speed (sdf):
    '''Find possible low outliers for speed by taking the lower bound as the given percentile
    Show the outliers and Output the lower bound calculated'''

    # Take the 5th percentile as the lower bound
    lower_bound = sdf.approxQuantile('speed', [0.02], 0.01)[0]

    # Show the possible lower outliers
    lower_df = sdf.filter(F.col('speed') < lower_bound)
    lower_df.show()

    # Output lower bound
    return lower_bound

def flag_weekend(sdf):
    '''Add a column to flag a day as weekend or not and output the resulted dataframe'''

    # Extract the day of the week using dayofweek
    # dayofweek returns 1 for Sunday and 7 for Saturday
    sdf = sdf.withColumn('day_of_week', F.dayofweek('tpep_pickup_datetime'))

    # Add a new column 'is_weekend' to flag if the day is Saturday (7) or Sunday (1)
    sdf = sdf.withColumn('is_weekend', ((F.col('day_of_week') == 1) | (F.col('day_of_week') == 7)).cast('boolean'))

    # Discard the 'day_of_week' column
    sdf = sdf.drop('day_of_week')

    return sdf

def flag_holiday (sdf, holiday_broadcast):
    '''Add a column to the dataframe to flag public holiday 
    and the scale of people affected by this holiday
    Output the flagged dataframe
    '''
    # Define a UDF to check if a date is a public holiday
    is_holiday_udf = F.udf(lambda date: date.strftime('%Y-%m-%d') in holiday_broadcast.value, BooleanType())

    # Define a UDF to get the holiday scale
    holiday_scale_udf = F.udf(lambda date: holiday_broadcast.value.get(date.strftime('%Y-%m-%d'), None), IntegerType())

    # Add the 'is_holiday' column to indicate if the pickup time is a public holiday
    df_with_holidays = sdf.withColumn('is_holiday', is_holiday_udf(F.to_date('tpep_pickup_datetime')))

    # Add the 'holiday_scale' column to indicate the scale of the holiday
    df_with_holidays = df_with_holidays.withColumn('holiday_scale', holiday_scale_udf(F.to_date('tpep_pickup_datetime')))
    
    # For non-holidays, scale is set to 0
    df_with_holidays = df_with_holidays.fillna({'holiday_scale': 0})
    
    return df_with_holidays

def preprocess_weather(sdf):
    '''Merge information of columns snow and snowdepth into one column
    Flag rainy days and select only relevant features
    Output processed dataframe'''
    
    # Merge information of snow and snowdepth
    sdf = sdf.withColumn(
    'snowdepth',
    F.when(F.col('snowdepth') < F.col('snow'), F.col('snow')).otherwise(F.col('snowdepth'))
)
    # Flag rainy days
    sdf = sdf.withColumn(
    'rain',
    F.when(F.col('conditions').contains('Rain'), 1).otherwise(0)
)
    # Select relevant features
    sdf = sdf.select(['datetime', 'temp', 'snowdepth', 'windgust', 'visibility', 'rain'])

    return sdf

def check_weather_data (sdf): 
    '''Output missing values and descriptive statistics in the given dataset'''
    print("Missing values:")
    missing_values = sdf.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) 
                                        for c in sdf.columns])
    missing_values.show()

    # Check range of time records
    print("datetime:\n\tLatest:", sdf.agg({"datetime": "max"}).collect()[0][0],
        "\n\tEarliest:", sdf.agg({"datetime": "min"}).collect()[0][0])
    
    # Descriptive statistics
    print("Descriptive statistics")
    description = sdf.select(["temp", "snowdepth", "windgust", "visibility"]).describe()
    description.show()

def filter_weather (set_name, sdf): 
    '''Remove instances with missing values and ensure the given dataframe
    is within correct timeframe
    Output the corrected dataframe'''

    # Drop any instances with missing values
    sdf = sdf.dropna()

    if set_name == 'train': 
        sdf = sdf.filter(
            # Filter for training data: 1st Dec 2022 to end of 31st May 2023
            (F.col("datetime").between("2022-12-01 00:00:00", "2023-05-31 23:59:59")))
    elif set_name == 'test':
        # Filter for testing data: 1st Jan 2024 to 31st May 2024
        sdf = sdf.filter(
            (F.col("datetime").between("2024-01-01 00:00:00", "2024-05-31 23:59:59")))
    else:
        raise ValueError("set_name should be either be 'train' or 'test'")

    return sdf
import os
import requests
from urllib.request import urlretrieve
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import IntegerType

def download_taxi(set_name, date):
    '''Download training and testing set from yellow taxi trip data 
    for the given dates and output them to the data/landing folders'''

    output_relative_dir = '../data/landing/'
    tlc_output_dir = output_relative_dir + set_name

    # Check if it exists as it makedir will raise an error if it does exist
    if not os.path.exists(output_relative_dir):
        os.makedirs(output_relative_dir)
    
    # now, for each type of data set we will need, we will create the paths
    if not os.path.exists(tlc_output_dir):
        os.makedirs(tlc_output_dir)
    
    # This is the URL template as of 08/2022
    URL_template = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_"

    for year in date.keys():
        for month in date[year]:
            # 0-fill i.e 1 -> 01, 2 -> 02, etc
            month = str(month).zfill(2) 
            print(f"Begin month {month}")

            # generate url
            url = f'{URL_template}{year}-{month}.parquet'
            # generate output location and filename
            output_dir = f"{tlc_output_dir}/{year}-{month}.parquet"
            # download
            urlretrieve(url, output_dir) 
            
            print(f"Completed month {month}")

def get_schema (sdf):
    '''Correct schema for the given dataframe by casting suitable data types 
    and output corrected schema'''
    sdf = sdf.withColumn("VendorID", sdf["VendorID"].cast(IntegerType())) \
       .withColumn("passenger_count", sdf["passenger_count"].cast(IntegerType())) \
       .withColumn("RatecodeID", sdf["RatecodeID"].cast(IntegerType())) \
       .withColumn("PULocationID", sdf["PULocationID"].cast(IntegerType())) \
       .withColumn("DOLocationID", sdf["DOLocationID"].cast(IntegerType())) \
       .withColumn("payment_type", sdf["payment_type"].cast(IntegerType())) \
       .withColumn("store_and_fwd_flag", 
                   F.when(F.col("store_and_fwd_flag") == "Y", True)
                    .when(F.col("store_and_fwd_flag") == "N", False)
                    .otherwise(None))  # Handles any unexpected values
    # Ensure consistent casing and ensure all columns exist
    consistent_col_casing = [F.col(col_name).alias(col_name.lower()) for col_name in sdf.columns]
    sdf = sdf.select(*consistent_col_casing)

    # this schema will be used for all other sets
    sdf_schema = sdf.schema
    return sdf_schema


def copy_schema(sdf, schema, output_path):
    '''Copy the given schema for the given set and output to the raw layer'''
    # copy schema
    sdf = sdf \
    .select([F.col(c).cast(schema[i].dataType) for i, c in enumerate(sdf.columns)])

    # Output to raw layer
    sdf \
    .coalesce(1) \
    .write \
    .mode('overwrite') \
    .parquet(output_path)

def download_weather(set_name):
    '''Download weather data for training and testing set'''

    # For train set
    if set_name == 'train':
        # URL of the CSV file
        url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/retrievebulkdataset?&key=JA3CCWYXBJX45H9RVREF562LW&taskId=b97034cf0db66fc29b340c0f8ccfa6a7&zip=false"

        # Download the file by sending an HTTP GET request
        response = requests.get(url)

        # Path where you want to save the file
        output_path = "../data/landing/train_weather.csv"

        # Write the content to a file
        with open(output_path, 'wb') as file:
            file.write(response.content)

    # Same for test set just different download link
    elif set_name == 'test':
        url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/retrievebulkdataset?&key=JA3CCWYXBJX45H9RVREF562LW&taskId=86f0f4cffd8ebd6e913d6e20c46bf1cd&zip=false"
        response = requests.get(url)
        output_path = "../data/landing/test_weather.csv"
        with open(output_path, 'wb') as file:
            file.write(response.content)

    else:
        raise ValueError("set_name should be either be 'train' or 'test'")
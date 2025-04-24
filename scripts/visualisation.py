from pyspark.sql import functions as F
import seaborn as sns
import matplotlib.pyplot as plt
import folium

def merge_datasets (taxi_data, weather_data):
    '''Merge taxi and weather datasets based on date and hour
    Output the merged dataframe'''

    # Extract day and hour from the timestamp column in taxi data
    taxi_data = taxi_data.withColumn('day', F.date_format(F.col('tpep_pickup_datetime'), 'yyyy-MM-dd')) \
            .withColumn('hour', F.hour(F.col('tpep_pickup_datetime')))

    # Extract day and hour from the timestamp column in weather
    weather_data = weather_data.withColumn('day', F.date_format(F.col('datetime'), 'yyyy-MM-dd')) \
            .withColumn('hour', F.hour(F.col('datetime')))

    # Perform the join on 'day' and 'hour'
    merged_df = taxi_data.join(weather_data, on=['day', 'hour'], how='inner')

    # Drop the 'datetime' column in weather data
    merged_df = merged_df.drop('datetime')
    
    return merged_df

def get_histogram(df, column_name, x_label, title):
    '''Plot a density histogram for the given column in the given dataset'''

    sns.histplot(df[column_name], kde=True, stat="density", bins=30)

    # Show the plot
    plt.xlabel(x_label)
    plt.ylabel('Density')
    plt.title(title)
    plt.show()

def plot_with_speed (sdf, bin_size, feature, bin_name, x_label, title):
    df_with_bins = sdf.withColumn(bin_name, 
                                (F.col(feature) / bin_size).cast('int') * bin_size)

    # Step 2: Calculate the average speed for each temperature interval
    avg_speed_df = df_with_bins.groupBy(bin_name).agg(F.avg('speed').alias('avg_speed'))

    # Step 3: Convert the result to a Pandas DataFrame
    avg_speed_pd = avg_speed_df.orderBy(bin_name).toPandas()

    # Step 4: Plot the average speed at different temperature intervals
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg_speed_pd, x=bin_name, y='avg_speed', marker='o')
    plt.xlabel(x_label)
    plt.ylabel('Average Speed (mph)')
    plt.title(title)
    plt.grid(True)
    plt.show()
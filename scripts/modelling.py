from pyspark.sql.functions import col, avg, count
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

def fit_pipeline (train_df, test_df):
    # Index categorical columns
    hour_indexer = StringIndexer(inputCol="hour", outputCol="hour_index", handleInvalid='keep')
    pulocationid_indexer = StringIndexer(inputCol="pulocationid", outputCol="pulocationid_index", handleInvalid='keep')
    dolocationid_indexer = StringIndexer(inputCol="dolocationid", outputCol="dolocationid_index", handleInvalid='keep')

    # One-hot encode the indexed columns
    hour_encoder = OneHotEncoder(inputCols=["hour_index"], outputCols=["hour_encoded"])
    pulocationid_encoder = OneHotEncoder(inputCols=["pulocationid_index"], outputCols=["pulocationid_encoded"])
    dolocationid_encoder = OneHotEncoder(inputCols=["dolocationid_index"], outputCols=["dolocationid_encoded"])

    # Assemble all features including the one-hot encoded columns and continuous features
    assembler = VectorAssembler(
        inputCols=["hour_encoded", "pulocationid_encoded", "dolocationid_encoded", "trip_distance", 
                "is_weekend", "holiday_scale", "temp", "windgust"],
        outputCol="assembled_features"
    )

    # Scale the assembled features
    scaler = StandardScaler(inputCol="assembled_features", outputCol="scaled_features")
    # Create and fit the pipeline
    pipeline = Pipeline(stages=[
    hour_indexer, pulocationid_indexer, dolocationid_indexer, # Indexing categorical variables
    hour_encoder, pulocationid_encoder, dolocationid_encoder, # One-hot encoding
    assembler,  # Assembling all features into a single vector
    scaler,     # Scaling the assembled features
    ])
    pipeline_model = pipeline.fit(train_df)

    # Transform the training and testing datasets
    train_prepared = pipeline_model.transform(train_df)
    test_prepared = pipeline_model.transform(test_df)

    return train_prepared, test_prepared

def make_predictions (train_prepared, test_prepared, model_name):
    '''Fit the given model and predict trip duration for the test set
    Compute predicted speed and return a prediction dataframe'''
    model = model_name.fit(train_prepared)
    predictions = model.transform(test_prepared)
    predictions = predictions.withColumn("predicted_speed", col("trip_distance") / (col("prediction")/60))

    return predictions

def predict_vs_actual (borough, lr_pred, rf_pred, date):
    '''Draw a graph to see the predicted average speed vs actual average speed
    every hour in a day'''

    # Filter the DataFrame to only include rows from the selected day
    lr_test_df_random_day = lr_pred.filter(col("day") == date)
    lr_speed_agg = lr_test_df_random_day.groupBy("hour").agg(
        avg("speed").alias("avg_actual_speed"),
        avg("predicted_speed").alias("avg_predicted_speed")
    )

    rf_test_df_random_day = rf_pred.filter(col("day") == date)
    rf_speed_agg = rf_test_df_random_day.groupBy("hour").agg(
        avg("predicted_speed").alias("avg_predicted_speed")
    )

    # Convert to Pandas for plotting
    lr_speed_agg_pd = lr_speed_agg.toPandas()
    rf_speed_agg_pd = rf_speed_agg.toPandas()

    # Plot the Results
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='hour', y='avg_actual_speed', data=lr_speed_agg_pd, marker='o', label='Actual Speed', color='blue')
    sns.lineplot(x='hour', y='avg_predicted_speed', data=lr_speed_agg_pd, marker='o', label='Predicted (Linear Regression)', color='red')
    sns.lineplot(x='hour', y='avg_predicted_speed', data=rf_speed_agg_pd, marker='o', label='Predicted (Random Forest)', color='green')

    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Speed (mph)')
    plt.title(f'Average Actual vs Predicted Speed on {date} in {borough}')
    plt.legend()
    plt.grid(True)
    plt.show()

def get_random_day (sdf):
    '''Output a random date with at least 4 trips in that day'''

    # Find days with at least 4 trips
    day_counts = sdf.groupBy("day").agg(count("*").alias("trip_count"))
    days_with_min_trips = day_counts.filter(col("trip_count") >= 4)
    days_list = days_with_min_trips.select("day").rdd.flatMap(lambda x: x).collect()

    # Ensure there are days with at least 4 trips
    if days_list:
        # Step 4: Select a random day from the list
        random_day = random.choice(days_list)
        return random_day
    else:
        print("No day has at least 5 trips in the DataFrame.")

def get_mae (predictions):
    '''Output the Mean Absolute Error for the predictions given'''
    evaluator = RegressionEvaluator(labelCol="trip_duration", predictionCol="prediction")
    mae = evaluator.setMetricName("mae").evaluate(predictions)
    return mae
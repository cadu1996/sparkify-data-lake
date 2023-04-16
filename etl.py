"""
This script is used to extract data from S3, process that data using Spark, and load the data back into S3.
This will be done in two steps:

    1. Load song_data and log_data from S3
    2. Process the data into analytics tables using Spark
    3. Load them back into S3
"""

import configparser
import os
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    monotonically_increasing_id,
    col,
    date_format,
    dayofmonth,
    hour,
    month,
    udf,
    weekofyear,
    year,
)

config = configparser.ConfigParser()
config.read("dl.cfg")

os.environ["AWS_ACCESS_KEY_ID"] = config.get("AWS", "AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = config.get("AWS", "AWS_SECRET_ACCESS_KEY")


def create_spark_session() -> SparkSession:
    """
    Create a Spark session

    Parameters:
        None

    Returns:
        SparkSession: Spark session
    """

    spark = SparkSession.builder.config(
        "spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.2"
    ).getOrCreate()
    
    return spark


def process_song_data(spark: SparkSession, input_data: str, output_data: str) -> None:
    """
    Process song data

    Parameters:
        spark (SparkSession): Spark session
        input_data (str): Input data path
        output_data (str): Output data path

    Returns:
        None
    """
    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*.json"

    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(
        col("song_id"),
        col("title"),
        col("artist_id"),
        col("year"),
        col("duration"),
        col("artist_name")
    ).distinct()

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(output_data + "songs_table", mode="overwrite")

    # extract columns to create artists table
    artists_table = df.select(
        col("artist_id"),
        col("artist_name").alias("name"),
        col("artist_location").alias("location"),
        col("artist_latitude").alias("latitude"),
        col("artist_longitude").alias("longitude"),
    ).distinct()

    # write artists table to parquet files
    artists_table.write.parquet(output_data + "artists_table", mode="overwrite")


def process_log_data(spark: SparkSession, input_data: str, output_data: str) -> None:
    """
    Process log data

    Parameters:
        spark (SparkSession): Spark session
        input_data (str): Input data path
        output_data (str): Output data path

    Returns:
        None
    """
    # get filepath to log data file
    log_data = input_data + "log_data/*.json"

    # read log data file
    df = spark.read.json(log_data)

    # filter by actions for song plays
    df = df.filter(df.page == "NextSong")

    # extract columns for users table
    users_table = df.select(
        col("userId").alias("user_id"),
        col("firstName").alias("first_name"),
        col("lastName").alias("last_name"),
        col("gender"),
        col("level")
    ).distinct()



    # write users table to parquet files
    users_table.write.parquet(output_data + "users_table", mode="overwrite")

    # create timestamp column from original timestamp column
    get_timestamp = udf(
        lambda x: datetime.fromtimestamp(x / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
    )
    df = df.withColumn("timestamp", get_timestamp(df.ts))

    # create datetime column from original timestamp column
    get_datetime = udf(
        lambda x: datetime.fromtimestamp(x / 1000.0).strftime("%Y-%m-%d")
    )
    df = df.withColumn("datetime", get_datetime(df.ts))

    # extract columns to create time table
    time_table = df.select(
        col("timestamp").alias("start_time"),
        hour(col("timestamp")).alias("hour"),
        dayofmonth(col("timestamp")).alias("day"),
        weekofyear(col("timestamp")).alias("week"),
        month(col("timestamp")).alias("month"),
        year(col("timestamp")).alias("year"),
        date_format(col("timestamp"), "E").alias("weekday"),
    ).distinct()

    # write time table to parquet files partitioned by year and month
    time_table.write.parquet(output_data + "time_table", mode="overwrite")

    # read in song data to use for songplays table
    song_df = spark.read.parquet(output_data + "songs_table")

    # extract columns from joined song and log datasets to create songplays table
    songplays_table = df.join(
        song_df, (col("song") == col("title")) & (col("artist") == col("artist_name"))
    ).select(
        monotonically_increasing_id().alias("songplay_id"),
        col("timestamp").alias("start_time"),
        col("userId").alias("user_id"),
        col("level"),
        col("song_id"),
        col("artist_id"),
        col("sessionId").alias("session_id"),
        col("location"),
        col("userAgent").alias("user_agent")
    )

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.parquet(
        output_data + "songplays_table", mode="overwrite"
    )


def main() -> None:
    """
    Main function
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://udacity-sparkify-data-lake/processed_data/"

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()

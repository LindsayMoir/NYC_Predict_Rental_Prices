#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Drop price outliers
    logger.info("Shape before min max price drop, %s", df.shape)
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    logger.info("Shape after min max price drop, %s", df.shape)

    # Drop the longitude and latitude outliers
    logger.info("Shape before lat long drop, %s", df.shape)
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    logger.info("Shape after lat long drop, %s", df.shape)

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    logger.info("last_review column converted to datetime")

    # Save the cleaned data in Weights & Biases
    artifact = wandb.Artifact(args.output_artifact, 
                              type = args.artifact_type,
                              description=args.artifact_description)
    df.to_csv("cleaned_sample.csv", index=False)  # Save the dataframe as a CSV file locally
    artifact.add_file("cleaned_sample.csv")  # Add the CSV file to the artifact
    run.log_artifact(artifact)  # Log the artifact to Weights & Biases
    logger.info("SUCCESS: Saved cleaned data to wandb: %s", args.output_artifact)

    # Make sure artifact is uploaded before this wandb run finishes
    artifact.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Get this dataset off of wandb",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="This is the output dataset",
        required=True
    )

    parser.add_argument(
        "--artifact_type", 
        type=str,
        help="Type of the output dataset",
        required=True
    )

    parser.add_argument(
        "--artifact_description", 
        type=str,
        help="Description of the output dataset",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=int,  
        help="Minimum value for the target variable (price)",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=int, 
        help="Maximum value for the target variable (price)",
        required=True
    )

    args = parser.parse_args()

    go(args)

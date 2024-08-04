#!/usr/bin/env python
"""
long_description [An example of a step using MLflow and Weights & Biases]: Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
from sklearn.model_selection import train_test_split
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="job_type [my_step]: train_val_test_split")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)
    logger.info("SUCCESS: Downloaded input artifact, %s:", args.input_artifact)  

    # Split the data into trainval and test datasets using train_test_split
    trainval_data, test_data = train_test_split(df, 
                                                test_size=args.test_size, 
                                                random_state=args.random_seed, 
                                                stratify=df[args.stratify_by])
    
    # Split the trainval_data into train and validation datasets using train_test_split
    train_data, val_data = train_test_split(trainval_data, 
                                            test_size=args.val_size, 
                                            random_state=args.random_seed, 
                                            stratify=trainval_data[args.stratify_by])
    logger.info("SUCCESS: Finished splitting data")

    # Save the datasets to disk
    train_data_path = "train_data.csv"
    val_data_path = "val_data.csv"
    test_data_path = "test_data.csv"

    train_data.to_csv(train_data_path, index=False)
    val_data.to_csv(val_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)

    # Log train_data to wandb
    train_artifact = wandb.Artifact(
        name=args.train_data,
        type="dataset",
        description="Train dataset",
    )
    train_artifact.add_file(train_data_path)
    run.log_artifact(train_artifact)
    logger.info("SUCCESS: Logged train_data to wandb")
    

    val_artifact = wandb.Artifact(
        name=args.val_data,
        type="dataset",
        description="Validation dataset"
    )
    val_artifact.add_file(val_data_path)
    run.log_artifact(val_artifact)
    logger.info("SUCCESS: Logged val_data to wandb")

    test_artifact = wandb.Artifact(
        name=args.test_data,
        type="dataset",
        description="Test dataset",
    )

    test_artifact.add_file(test_data_path)
    run.log_artifact(test_artifact)
    logger.info("SUCCESS: Logged test_data to wandb")

    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step will split the dataset into \
                                     train, validation and test sets")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Get this dataset off of wandb. This is the original sample dataset",
        required=True
    )

    parser.add_argument(
        "--train_data", 
        type=str,
        help="This is the train  dataset",
        required=True
    )

    parser.add_argument(
        "--val_data", 
        type=str,
        help="This is the validation  dataset",
        required=True
    )

    parser.add_argument(
        "--test_data", 
        type=str,
        help="This is the test dataset",
        required=True
    )

    parser.add_argument(
        "--test_size", 
        type=float,
        help="Size of the test set in terms of percentage of the original dataset",
        required=True
    )

    parser.add_argument(
        "--val_size", 
        type=float,  
        help="Size of the validation set in terms of percentage of the original dataset",
        required=True
    )

    parser.add_argument(
        "--random_seed", 
        type=int, 
        help="Random seed for reproducibility",
        required=True
    )

    parser.add_argument(
        "--stratify_by", 
        type=str, 
        help="Column to stratify the sample on",
        required=True
    )

    args = parser.parse_args()

    go(args)

import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):

    expected_columns = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_columns) == list(these_columns)
    print("expected_columns test completed")


def test_neighborhood_names(data):

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)
    print("neighbourhood_names test completed")


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0
    print("proper_boundaries test completed")

def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold
    print("similar_neigh_distrib test completed")


########################################################
# Implement here test_row_count and test_price_range   #
########################################################
def test_row_count(data):
    """
    Test that the number of rows in the dataset is at least min_rows
    """
    assert 15000 < data.shape[0] < 100000
    print("row_count test completed")


def test_price_range(data, min_price, max_price):
    """
    Test that the price range is within the expected range
    """
    assert data['price'].between(min_price, max_price).all()
    print("price_range test completed")
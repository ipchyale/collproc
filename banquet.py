import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.expanduser("~")+"/ivpy/src")
from ivpy import *
from ivpy.cluster import cluster
import warnings
warnings.filterwarnings('ignore')

"""
This module contains functions for applying the Banquet Table constraint to a DBSCAN clustering.
The Banquet Table requires a collection of items and a reference set of items. Items in the 
collection are "attendees" of a banquet, and items in the reference set are "guests" of the
attendees. A reference set item R is a guest of attendee A if R is closer to A than distance d,
where d is chosen by the user. The Banquet Table requires, of any two attendees A and B, that
if A and B "sit at the same table" (are in the same cluster), then A and B must have at least one
guest in common.

Attendees with no guests are called "lonelies". Attendees with guests but no allowable tablemates 
are called "outsiders". Neither lonelies nor outsiders are considered in the DBSCAN clustering, 
but we keep them in the collection frame.

We choose the DBSCAN clustering with the minimum value of epsilon that satisfies the Banquet Table
constraint. Other DBSCAN parameters are chosen by the user, and we usually set min_samples=2.
"""

def get_guest_lists(collection_frame, reference_frame, feature_columns, d):
    """
    Vectorized computation of guest lists using broadcasting
    """
    collection_array = collection_frame[feature_columns].values
    reference_array = reference_frame[feature_columns].values
    distances = np.linalg.norm(collection_array[:, np.newaxis] - reference_array, axis=2)
    reference_indices = reference_frame.index.values
    guest_lists = [list(reference_indices[row < d]) for row in distances]
    return pd.Series(guest_lists, index=collection_frame.index)

def get_allowable_tablemates_lists(guest_lists):
    """
    Returns a pandas Series with the same indices as guest_lists, where each element is a list of
    allowable tablemates for that collection item. A collection item A and a collection item B are
    allowable tablemates if A and B have at least one guest in common. Thus, A and B are allowable
    tablemates if the intersection of their guest lists is nonempty.

    The allowable tablemates list for a collection item does not include the collection item itself;
    if we leave them in, "outsiders" are impossible.
    """

    guest_sets = guest_lists.apply(set)
    indices = guest_lists.index
    allowable_tablemates_lists = [
        [idx for idx in indices if idx != collection_idx and guest_sets[collection_idx].intersection(guest_sets[idx])]
        for collection_idx in indices
    ]
    return pd.Series(allowable_tablemates_lists, index=indices)

def is_tabling_allowable(tabling, allowable_tablemates_lists):
    """
    `tabling` is a pandas Series of cluster labels for the collection items. Returns True if the
    clustering satisfies the Banquet Table constraint, and False otherwise. Some cluster labels
    will be -1, indicating that the collection item is placed in the "noise" cluster by DBSCAN.
    Noise items are not subject to the Banquet Table constraint, because they are not tabled.
    """
     
    tables = tabling.unique()
    tables = tables[tables != -1]

    for table in tables:
        table_members_set = set(tabling[tabling == table].index)
        for member in table_members_set:
            # Remove the current member from the table members set for this check
            candidate_tablemates = table_members_set - {member}
            if not candidate_tablemates.issubset(allowable_tablemates_lists.loc[member]):
                return False

    return True

def banquet_tabling(collection_frame, reference_frame, feature_columns, d, eps=0.164, eps_increment=0.0001):
    """
    Returns a pandas Series (with `collection_frame` indices) of cluster labels for the collection 
    items, where the clustering satisfies the Banquet Table constraint. Lonelies and outsiders are
    not included in the clustering.
    """

    guest_lists = get_guest_lists(collection_frame, reference_frame, feature_columns, d)
    allowable_tablemates_lists = get_allowable_tablemates_lists(guest_lists)

    lonelies = guest_lists[guest_lists.str.len() == 0].index
    outsiders = allowable_tablemates_lists[allowable_tablemates_lists.str.len() == 0].index.difference(lonelies)
    attendees = collection_frame.index.difference(lonelies.union(outsiders))

    tabling = None
    while True:
        tabling = cluster(collection_frame[feature_columns].loc[attendees], method='dbscan', eps=eps, min_samples=2)
        if is_tabling_allowable(tabling, allowable_tablemates_lists):
            break
        eps -= eps_increment

    tabling_with_lonelies_and_outsiders = pd.Series(index=collection_frame.index, dtype=object)
    tabling_with_lonelies_and_outsiders[attendees] = tabling
    tabling_with_lonelies_and_outsiders[lonelies] = 'lonely'
    tabling_with_lonelies_and_outsiders[outsiders] = 'outsider'
    
    return tabling_with_lonelies_and_outsiders
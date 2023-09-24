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

def get_distance(collection_frame,collection_index,reference_frame,reference_index,feature_columns):
    """
    Returns Euclidean distance between reference item and collection item, 
    in feature space with axes 'feature_columns'.
    """
    
    collection_array = np.array(collection_frame[feature_columns].loc[collection_index])
    reference_array = np.array(reference_frame[feature_columns].loc[reference_index])

    return np.linalg.norm(collection_array-reference_array)

def get_guest_list(collection_frame,reference_frame,collection_index,feature_columns,d):
    """
    Runs get_distance() on every reference item for a single collection item. Returns a 
    list of indices from reference_frame of reference items that are "guests" of the 
    collection item, where a "guest" is a reference item that is closer to the collection 
    item than distance d.
    """

    guest_list = []
    for reference_index in reference_frame.index:
        if get_distance(collection_frame,collection_index,reference_frame,reference_index,feature_columns) < d:
            guest_list.append(reference_index)
    
    return guest_list
    
def get_guest_lists(collection_frame,reference_frame,feature_columns,d):
    """
    Runs get_guest_list() on every collection item. Returns a pandas Series with the same
    indices as collection_frame, where each element is a list of indices from reference_frame
    """

    guest_lists = []
    for collection_index in collection_frame.index:
        guest_lists.append(get_guest_list(collection_frame,reference_frame,collection_index,feature_columns,d))
    
    return pd.Series(guest_lists,index=collection_frame.index)

def get_allowable_tablemates_lists(guest_lists):
    """
    Returns a pandas Series with the same indices as guest_lists, where each element is a list of
    allowable tablemates for that collection item. A collection item A and a collection item B are
    allowable tablemates if A and B have at least one guest in common. Thus, A and B are allowable
    tablemates if the intersection of their guest lists is nonempty.

    We do not include a collection item in its own list of allowable tablemates.
    """
    
    allowable_tablemates_lists = []
    for collection_index in guest_lists.index:
        allowable_tablemates = []
        for other_collection_index in guest_lists.index:
            if collection_index != other_collection_index:
                if len(set(guest_lists[collection_index]).intersection(set(guest_lists[other_collection_index]))) > 0:
                    allowable_tablemates.append(other_collection_index)
        allowable_tablemates_lists.append(allowable_tablemates)

    return pd.Series(allowable_tablemates_lists,index=guest_lists.index)

def is_tabling_allowable(tabling,allowable_tablemates_lists):
    """
    `tabling` is a pandas Series of cluster labels for the collection items. Returns True if the
    clustering satisfies the Banquet Table constraint, and False otherwise. Some cluster labels
    will be -1, indicating that the collection item is placed in the "noise" cluster by DBSCAN.
    Noise items are not subject to the Banquet Table constraint, because they are not tabled.

    Some cluster labels will be null, because the collection item is a lonely or an outsider and 
    is not entered into DBSCAN at all. We ignore these items when checking the Banquet Table
    constraint.
    """

    # Get a list of all clusters that are neither noise nor null
    tabling = tabling.loc[tabling.notnull()]
    tabling = tabling.loc[tabling != -1]
    tables = tabling.unique()

    # Check the Banquet Table constraint for each table
    for table in tables:
        table_members = tabling.loc[tabling == table].index
        for table_member in table_members:
            allowable_tablemates = allowable_tablemates_lists[table_member]
            if len(set(table_members).intersection(set(allowable_tablemates))) == 0:
                return False
            
    return True

def banquet_tabling(collection_frame,reference_frame,feature_columns,d,eps=0.164,eps_increment=0.0001):
    """
    Returns a pandas Series (with `collection_frame` indices) of cluster labels for the collection 
    items, where the clustering satisfies the Banquet Table constraint. Lonelies and outsiders are
    not included in the clustering.
    """

    guest_lists = get_guest_lists(collection_frame,reference_frame,feature_columns,d)
    allowable_tablemates_lists = get_allowable_tablemates_lists(guest_lists)

    lonelies = guest_lists.loc[guest_lists.apply(len) == 0].index
    outsiders = allowable_tablemates_lists.loc[allowable_tablemates_lists.apply(len) == 0].index
    attendees = collection_frame.index.difference(lonelies.union(outsiders))

    tabling = cluster(collection_frame[feature_columns].loc[attendees],
                      method='dbscan',eps=eps,min_samples=2)

    while is_tabling_allowable(tabling,allowable_tablemates_lists) == False:
        eps -= eps_increment
        tabling = cluster(collection_frame[feature_columns].loc[attendees],
                      method='dbscan',eps=eps,min_samples=2)
        
    tabling_with_lonelies_and_outsiders = pd.Series(index=collection_frame.index)
    tabling_with_lonelies_and_outsiders.loc[attendees] = tabling
    tabling_with_lonelies_and_outsiders.loc[lonelies] = 'lonely'
    tabling_with_lonelies_and_outsiders.loc[outsiders] = 'outsider'
        
    return tabling_with_lonelies_and_outsiders
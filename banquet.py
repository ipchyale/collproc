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

    metric_tabling = tabling.loc[tabling != -1]
    table_sharing_ratios = (intra_table_sharing_ratio(metric_tabling, guest_lists),
                            inter_table_sharing_ratio(metric_tabling, guest_lists))

    metrics = {
        'd': d,
        'tableable_percentage': compute_tableable_percentage(tabling_with_lonelies_and_outsiders),
        'noise_percentage': compute_noise_percentage(tabling),
        'intra_table_sharing_ratio': table_sharing_ratios[0],
        'inter_table_sharing_ratio': table_sharing_ratios[1],
        'guest_sharing_differential': guest_sharing_differential(table_sharing_ratios[0], table_sharing_ratios[1]),
        'overlap_coefficient': overlap_coefficient(metric_tabling, guest_lists),
        'jaccard_index': jaccard_index(metric_tabling, guest_lists),
        'unique_vs_shared_guests': unique_vs_shared_guests(metric_tabling, guest_lists)
    }

    return tabling_with_lonelies_and_outsiders, metrics

#------------------------------------------------------------------------------

def compute_tableable_percentage(tabling):
    total = len(tabling)
    lonelies_and_outsiders = (tabling == 'lonely').sum() + (tabling == 'outsider').sum()
    return (total - lonelies_and_outsiders) / total

def compute_noise_percentage(tabling):
    return (tabling == -1).sum() / len(tabling)

# Guest-sharing metrics

def intra_table_sharing_ratio(tabling, guest_lists):

    """
    This measures the extent to which members within a table share guests.
    """

    tables = tabling.unique()
    
    ratios = []

    for table in tables:
        table_members = tabling[tabling == table].index
        all_guests = [guest_lists.loc[member] for member in table_members]
        all_guests = [item for sublist in all_guests for item in sublist]
        guest_counts = pd.Series(all_guests).value_counts()
        total_guests = len(guest_counts)
        shared_guests = (guest_counts > 1).sum()

        ratio = shared_guests / total_guests if total_guests else 0
        ratios.append(ratio)

    return np.mean(ratios)

def inter_table_sharing_ratio(tabling, guest_lists):

    """
    This measures the extent to which tables share guests. So, rather than 
    count every time two members at different tables share a guest, we find the 
    union of all guest lists for each table and treat each table as a single entity.
    """

    tables = tabling.unique()

    table_guest_lists = []
    for table in tables:
        table_members = tabling[tabling == table].index
        table_guest_list = list(set.union(*[set(guest_lists[member]) for member in table_members]))
        table_guest_lists.append(table_guest_list)

    all_guests = [item for sublist in table_guest_lists for item in sublist]
    guest_counts = pd.Series(all_guests).value_counts()
    total_guests = len(guest_counts)
    shared_guests = (guest_counts > 1).sum()

    ratio = shared_guests / total_guests if total_guests else 0

    return ratio

def guest_sharing_differential(intra_table_sharing_ratio, inter_table_sharing_ratio):

    """
    This measures the difference between intra and inter-table guest sharing.
    """

    return intra_table_sharing_ratio - inter_table_sharing_ratio

def overlap_coefficient(tabling, guest_lists):

    """
    For any two tables: the size of the intersection of their guest 
    lists divided by the size of the smaller of the two guest lists.
    """

    tables = tabling.unique()

    table_guest_sets = []
    for table in tables:
        table_members = tabling[tabling == table].index
        table_guest_set = set.union(*[set(guest_lists[member]) for member in table_members])
        table_guest_sets.append(table_guest_set)

    # Compute the overlap coefficient for each pair of tables
    coefficients = []
    for i in range(len(table_guest_sets)):
        for j in range(i+1, len(table_guest_sets)):
            intersection = len(table_guest_sets[i].intersection(table_guest_sets[j]))
            min_size = min(len(table_guest_sets[i]), len(table_guest_sets[j]))
            coefficient = intersection / min_size
            coefficients.append(coefficient)

    return np.mean(coefficients)

def jaccard_index(tabling, guest_lists):

    """
    For any two tables: the size of the intersection of their guest 
    lists divided by the size of the union of their guest lists.
    """

    tables = tabling.unique()

    table_guest_sets = []
    for table in tables:
        table_members = tabling[tabling == table].index
        table_guest_set = set.union(*[set(guest_lists[member]) for member in table_members])
        table_guest_sets.append(table_guest_set)

    # Compute the Jaccard index for each pair of tables
    indices = []
    for i in range(len(table_guest_sets)):
        for j in range(i+1, len(table_guest_sets)):
            intersection = len(table_guest_sets[i].intersection(table_guest_sets[j]))
            union = len(table_guest_sets[i].union(table_guest_sets[j]))
            index = intersection / union
            indices.append(index)

    return np.mean(indices)

def unique_vs_shared_guests(tabling, guest_lists):

    """
    This metric counts the total number of unique guests across all tables and compares 
    it to the total number of guests shared more than once by any two members, whether
    they are at the same table or not.
    """

    tables = tabling.unique()

    all_guests = set()
    shared_guests = set()

    # Gather all guests and their occurrences
    guest_occurrences = {}
    for table in tables:
        table_members = tabling[tabling == table].index
        table_guests = set.union(*[set(guest_lists[member]) for member in table_members])
        all_guests.update(table_guests)

        for guest in table_guests:
            guest_occurrences[guest] = guest_occurrences.get(guest, 0) + 1

    # Determine which guests are shared across tables
    for guest, count in guest_occurrences.items():
        if count > 1:
            shared_guests.add(guest)

    total_unique_guests = len(all_guests)
    total_shared_guests = len(shared_guests)

    ratio = total_shared_guests / total_unique_guests if total_unique_guests else 0

    return ratio

#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

def find_inflection_points(mf, diff_threshold=0.1):
    """
    Returns a list of crucial values of d, where metrics make big jumps.
    """

    # Find several inflection points for each metric
    inflection_points = []
    for col in [item for item in mf.columns if item != 'd']:
        for i in range(1, len(mf)):
            if mf[col].iloc[i] - mf[col].iloc[i-1] > diff_threshold:
                d_value = round(mf.d.iloc[i], 3)
                inflection_points.append(d_value)

    # return all inflection points appearing more than once
    return [item for item in inflection_points if inflection_points.count(item) > 1]    

def plot_metrics(metrics_list,inflection_points=True,diff_threshold=0.1):
    
    mf = pd.DataFrame(metrics_list)
    
    plt.figure(figsize=(15,10))
    plt.gcf().set_facecolor('gainsboro')
    plt.gca().set_facecolor('gainsboro')
    plt.grid(True, color='white')

    # Add labels, title, and legend with thicker lines
    plt.xlabel('d values')
    plt.ylabel('Metric values')
    plt.title('Tabling metrics over different d values')

    for col in [item for item in mf.columns if item != 'd']:
        plt.plot(mf.d,mf[col],label=col,linewidth=5)
    
    # add vertical lines and labels for inflection points
    if inflection_points:
        inflection_points = find_inflection_points(mf,diff_threshold=diff_threshold)
        for point in inflection_points:
            plt.axvline(x=point, color='black', linestyle='--', linewidth=2)
            plt.text(point,0.5,str(point),rotation=90,fontsize=12)

    plt.legend(loc='best',handlelength=1)

    return plt.gcf()

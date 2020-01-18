# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import skew

def label_normalizer(centers_, labels_):
    """
    Normalizes label annotation according to the left-most cluster 
    to the right-most in terms of position in the vector space
    
    Args:
        centers_ (np.array): Center of each micro_cluster
        labels_ (np.array): Labels of each micro_cluster
    
    Returns:
        (np.array) Labels normalized
    """

    if len(centers_) > 0 and len(labels_) > 0:

        # ordered = [(c, x) for c,x in sorted(zip(old_centers, old_labels))]
        value = 0
        maps = {}
        
        # for _, l in sorted(zip(list(centers_), list(labels_))):
        for l in labels_[np.lexsort(np.transpose(centers_)[::-1])]:
            if l not in maps:
                maps[l] = value
                value += 1

        return np.vectorize(maps.get)(labels_)
    return None


def micro_cluster_predict_norm(model, X):
    """
    Model prediction based on normalized micro_cluster labels
    
    Args:
        model (sklearn): BIRCH model
        X (np.array): data points to be predicted
    
    Returns:
        (np.array) normalized predicted labels
    """
    micro_clusters = model.transform(X).argmin(axis=1)
    norm_labels = label_normalizer(model.subcluster_centers_, model.subcluster_labels_)
    labels = [norm_labels[m] for m in micro_clusters]
    return np.array(labels)


def keys_values_with_max_value(dict):
    """
    Returns a (key, value) pair of the key with the max value in the dict.
    Source: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
    """
    return [(key, val) for m in [max(dict.values())] 
            for key, val in dict.items() if val == m] 


def get_cf_subclusters(model):
    """
    Returns subclusters in dictionary form with data in the schema
    Cluster Feature with 'n_samples', 'linear_sum', 'squared_sum', 
    'centroid', 'sq_norm', 'radius' e 'label'

    Args: 
        model (sklearn): BIRCH model

    Returns:
        (pd.DataFrame) with rows describing subclusters
    """
    resp = []
    
    labels = label_normalizer(model.subcluster_centers_, model.subcluster_labels_)
    
    i = 0
    for leaf in model._get_leaves():
        for x in leaf.subclusters_:
            resp.append({
                "n_samples": x.n_samples_,
                "linear_sum": x.linear_sum_,
                "squared_sum": x.squared_sum_,
                "centroid": x.centroid_,
                "sq_norm": x.sq_norm_,
                "radius": x.radius,
                "label": labels[i]
            })
            i += 1
    
    return(pd.DataFrame(resp))


def get_array_ordered_by(lista, positions):
    """
        Retorna um array ordenado em relação à posições
        lidando com valores não existentes
    """
    r = []
    
    for pos in positions:
        if pos < len(lista):
            r.append(lista[pos])
        else:
            r.append(-1)
    
    return np.array(r)


def compara_subclusters(cf_subclusters_1, cf_subclusters_2, diffs=False):
    """
    Compare subclusters 1 and 2 and returns metrics evaluating them

    Args:
        cf_subclusters_1 (pd.DataFrame): subclusters from 'get_cf_subclusters'
        cf_subclusters_2 (pd.DataFrame): subclusters from 'get_cf_subclusters' at next timestep
    
    Returns:
        (dict): Metrics comparing two clusterings 
    """
    centroid_dimensions = cf_subclusters_1.centroid.iloc[0].shape[0]
    
    centroids_1 = np.concatenate(cf_subclusters_1.centroid.values).reshape(-1, centroid_dimensions)
    centroids_2 = np.concatenate(cf_subclusters_2.centroid.values).reshape(-1, centroid_dimensions)
    
    labels_1 = np.array(cf_subclusters_1.label.values)
    labels_2 = np.array(cf_subclusters_2.label.values)
    
    distances = distance.cdist(centroids_1, centroids_2)
    min_distances = distances.min(axis=1)
    min_distances_i = distances.argmin(axis=1)
    
    r = {
        "avg_min_distances": min_distances.mean(),
        "std_min_distances": min_distances.std(),
        "not_changed": (min_distances == 0).sum()/len(min_distances),
        "new_subclusters": len(cf_subclusters_1) - len(cf_subclusters_2),
        "num_splits": (np.unique(min_distances_i, return_counts=True)[1] > 1).sum(),
        "skew_change": skew(centroids_2, axis=None) - skew(centroids_1, axis=None),
        "labels_changed": np.sum(labels_2 != get_array_ordered_by(labels_1, min_distances_i))/len(labels_2),
        
    }
    
    for group in np.unique(labels_2):
        g1 = cf_subclusters_1[cf_subclusters_1.label == group]
        g2 = cf_subclusters_2[cf_subclusters_2.label == group]
        
        if len(g1)>0 and len(g2) > 0: 
            r[str(group) + "_size"] = len(g2)           
            r[str(group) + "_skew"] = skew(np.concatenate(g2.centroid.values), axis=None)             
            r[str(group) + "_mean"] = np.mean(g2.centroid.values.mean())
            r[str(group) + "_radius"] = g2.radius.mean()
            r[str(group) + "_sq_norm"] = g2.sq_norm.mean()            
            r[str(group) + "_squared_sum"] = g2.squared_sum.mean()            
            r[str(group) + "_n_samples"] = g2.n_samples.mean()
            
            if diffs:
                r[str(group) + "_size_diff"] = len(g2) - len(g1)
                r[str(group) + "_skew_diff"] = skew(np.concatenate(g2.centroid.values), axis=None) - skew(np.concatenate(g1.centroid.values), axis=None)
                r[str(group) + "_mean_diff"] = distance.euclidean(g2.centroid.mean(axis=0), g1.centroid.mean(axis=0))
                r[str(group) + "_radius_diff"] = g2.radius.mean() - g1.radius.mean()
                r[str(group) + "_sq_norm_diff"] = g2.sq_norm.mean() - g1.sq_norm.mean()
                r[str(group) + "_squared_sum_diff"] = g2.squared_sum.mean() - g1.squared_sum.mean()
                r[str(group) + "_n_samples_diff"] = g2.n_samples.mean() - g1.n_samples.mean()
    
    return r
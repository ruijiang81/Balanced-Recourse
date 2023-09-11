from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np 
import pandas as pd 
import seaborn as sns 

def get_reward(ml_model, true_cf_received, rewardtype = 'validation'):
    # compute the reward for each user
    labels_predicted  = ml_model.predict(true_cf_received)
    if rewardtype == 'validation':
        reward = sum(labels_predicted>0.5)
    return reward

def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    
    # Set the font to be serif, rather than sans
    sns.set(font='serif')
    
    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Palatino", "serif"]
    })


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def encode_constraint(cf, negative, sens_name):
    for sen_name in sens_name:
        cf[sen_name] = negative[sen_name]
    return cf

def invalidation(cf, ml_model):
    inv = ml_model.predict(cf)
    return np.mean(inv < 0.5)

def centralization(sens, features, radius = None):
    if sum(sens==0) >  sum(sens==1):
        sensid = 1
    else:
        sensid = 0
    minority = features[sens==sensid]
    majority = features[sens==1-sensid]
    distance_cross = pairwise_distances(minority, majority, metric='euclidean')
    distance_in = pairwise_distances(minority, minority, metric='euclidean')
    
    if radius is None:
        radius = np.quantile(distance_in, 0.1) # syn 0.1 
    central_region =  ((np.sum(distance_in < radius, 1)) / ((np.sum(distance_cross < radius, 1))+(np.sum(distance_in < radius, 1)))) 
    central_region = central_region > 0.5
    num_central = sum([i==sensid and j ==1 for i,j in zip(sens, central_region)]) * 1. 
    ci = num_central / sum(sens == sensid)
    return ci, radius 


def avg_proximity2(sens, features, origin_features = None):
    if sum(sens==0) >  sum(sens==1):
        sensid = 1
    else:
        sensid = 0
    C_ij = pairwise_distances(features, metric='l2')
    C_ij = np.exp(-C_ij)
    a = 0 
    m0 = sum(sens == sensid)
    m1 = sum(sens == 1-sensid)
    num_classes = C_ij.shape[0]
    a = np.sum((C_ij.T * np.array(sens==sensid)).T * np.array(1-(sens==sensid))) / (m0 * m1)
    return a




def atkinson(sens, features, origin_features = None, beta = 0.5):
    # how to define a neighborhood 
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances
    if sum(sens==0) >  sum(sens==1):
        sensid = 1
    else:
        sensid = 0
    nn = 30
    cluster = KMeans(n_clusters = nn, random_state = 0)
    if origin_features is not None:
        cluster.fit(origin_features)
    neighborhood = cluster.predict(features)

    num_classes = nn
    P = sum(sens==sensid) / len(sens)
    T = len(sens)
    ak = 0
    for i in range(num_classes):
        ti = sum(neighborhood==i)
        if ti == 0:
            continue
        mi = sum(sens[neighborhood==i] == sensid)
        pi = mi / ti 
        aki = (1-pi) ** (1-beta) * (pi ** beta) * ti / (T * P)
        ak += aki
    ak = 1 - P/(1-P) * np.abs(ak)**(1/(1-beta))
    return ak


def recourse_cost(origin_features, new_features):
    return np.sqrt(((origin_features - new_features)**2).sum(1)).mean()

def fairness_cost(origin_features, new_features, sens):
    rec1 = np.sqrt(((origin_features[sens==1] - new_features[sens==1])**2).sum(1)).mean()
    rec2 = np.sqrt(((origin_features[sens==0] - new_features[sens==0])**2).sum(1)).mean()
    print(rec1)
    print(rec2)
    print(np.abs(rec1-rec2))
    print(origin_features.shape)
    print(sens.shape)
    return np.abs(rec1-rec2)

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from carla.evaluation import remove_nans
from carla.evaluation.api import Evaluation

def ynn(factuals, cf, ml_model, num = 5):
    factuals = ml_model.get_ordered_features(ml_model.data.df)
    cf = cf[factuals.columns]
    cf = cf.fillna(cf.mean(skipna=False))
    cf = cf.fillna(0)
    number_of_diff_labels = 0
    nbrs = NearestNeighbors(n_neighbors=num).fit(factuals.values)
    for i, row in cf.iterrows():
        knn = nbrs.kneighbors(
            row.values.reshape((1, -1)), num, return_distance=False
        )[0]
        for idx in knn:
            neighbour = factuals.iloc[idx]
            neighbour = neighbour.values.reshape((1, -1))
            neighbour_label = np.argmax(ml_model.predict_proba(neighbour))
            number_of_diff_labels += np.abs(1 - neighbour_label)

    return 1 - (1 / (len(cf) * num)) * number_of_diff_labels

def closeness(cf, positive_points):
    cf = cf.fillna(cf.mean(skipna=False))
    cf = cf.fillna(0)    
    distance = pairwise_distances(cf, positive_points)
    distance = distance.mean()
    return distance 
#!/usr/bin/env python3

import warnings
import os, glob
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import seaborn as sns


class CreateLDAModel():
    def __init__(self):
        self.concatenated_sources = None
        self.counter = None
        self.transformed_features = None
        self.best_estimator = None
        self.params_associated_to_best_estimator = None
        self.transformed_features_via_lda = None

    def concatenate_sources(self):
        tmp_filelist = glob.glob(os.getcwd() + '/data/*.csv')
        parse_data = lambda x: pd.read_csv(x, index_col=None, header=0).drop(columns = 'Unnamed: 0')
        tmp_parsed_data = [parse_data(file) for file in tmp_filelist]
        tmp_parsed_data = pd.concat(tmp_parsed_data)
        tmp_parsed_data.columns = ['artist', 'genre', 'song', 'lyrics']
        self.concatenated_sources = tmp_parsed_data

    def transform_to_sparse_matrix(self, stop_words, max_features):
        self.counter = CountVectorizer(stop_words=stop_words, max_features=max_features)
        self.transformed_features = self.counter.fit_transform(self.concatenated_sources['lyrics'].values)

    def infer_best_hyperparams_for_lda(self, hyperparams):
        tmp_grid_search_model = GridSearchCV(
            LatentDirichletAllocation(learning_method='online'),
            param_grid=hyperparams
        )
        tmp_grid_search_model.fit(self.transformed_features)
        self.best_estimator = tmp_grid_search_model.best_estimator_

    def train_specific_model(self, n_components,learning_decay):
        self.best_estimator = LatentDirichletAllocation(learning_method='online',
                                                        n_components = n_components,
                                                        learning_decay=learning_decay).fit(self.transformed_features)

    def serialize_results(self, filename):
        with open(f"./pickles/{filename}_train.pkl", 'wb') as trained_model:
            joblib.dump(self.best_estimator, trained_model)
        with open(f"./pickles/{filename}_features.pkl", 'wb') as transformed_features:
            joblib.dump(self.transformed_features,transformed_features)
        with open(f"./pickles/{filename}_counter.pkl", "wb") as counter_vect:
            joblib.dump(self.counter, counter_vect)





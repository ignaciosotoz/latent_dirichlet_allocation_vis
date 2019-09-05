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


class DiagnosisLDA():
    def __init__(self, original_data, model, features, counter):
        self.best_estimator = model
        self.transformed_features = features
        self.counter = counter
        self.concatenated_sources = original_data
        self.transformed_features_via_lda = None
        self.topics = None

    def infer_topics_on_model(self, words_per_topic=15):
        tmp_dictionary_holder = {}
        for topic_id, topic_name in enumerate(self.best_estimator.components_):
            tmp_words_per_topic = [self.counter.get_feature_names()[topic] for topic in topic_name.argsort()[:-words_per_topic - 1: -1]]
            tmp_dictionary_holder[f"{topic_id + 1}"] = tmp_words_per_topic
        self.topics = tmp_dictionary_holder

    def infer_probability_mixture(self):
        tmp_pr_mixture = self.best_estimator.transform(self.transformed_features)
        self.transformed_features_via_lda = tmp_pr_mixture
        self.transformed_features_via_lda = pd.DataFrame(
            np.round(self.transformed_features_via_lda, 3),
            index = self.concatenated_sources.index
        )
        label_columns = list(map(lambda x: f"Tópico: {x}", range(1, self.best_estimator.n_components + 1)))
        tmp_pr_mixture = pd.DataFrame(np.round(tmp_pr_mixture, 3), columns=label_columns, index=self.concatenated_sources.index)
        self.transformed_features_via_lda.columns = label_columns
        self.transformed_features_via_lda = pd.concat([self.concatenated_sources,
                                                       self.transformed_features_via_lda],
                                                       axis = 1)

        self.transformed_features_via_lda['highest_topic'] = np.argmax(tmp_pr_mixture.values, axis=1) + 1




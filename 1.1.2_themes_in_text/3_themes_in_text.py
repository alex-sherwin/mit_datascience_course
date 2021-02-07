#!/usr/bin/env python3

import json
from nltk import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

# we will create a plot with 5 rows and 2 columns:
# [bar][bar][bar][bar][bar]  <-- 5 topics
# [          bar          ]  <-- lab breakdown by topic

mit_labs = set(['CSAIL', 'LIDS', 'MTL', 'RLE'])

facultyDatas = json.loads(open('data/faculty', 'r').read())

# collect all abstracts
abstracts = []

# track a list of labs for each article
abstractLabs = []

# LDA algo tweakable params
n_features = 1000
n_components = 5
n_top_words = 10


def processAbstractsForFacultyMember(facultyMember):
    facultyName = facultyMember['name']
    labs = facultyMember['labs']
    articleData = json.loads(
        open('data/articles/' + facultyName, 'r').read()
    )
    for article in articleData['articles']:
        abstracts.append(article['abstract'])
        abstractLabs.append(labs)

    return abstracts


def plotTopWords(ldaModel, feature_names, n_top_words):
    for topic_idx, topic in enumerate(ldaModel.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        subplot_offset = topic_idx * 2

        plt.subplot(2, 9, subplot_offset + 1,
                    title='Topic ' + str(topic_idx+1))
        bar = plt.barh(top_features, weights)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.invert_yaxis()
        plt.tick_params(axis='both', which='major', labelsize=10, pad=10)


def plotTopicBreakdown(ldaModel, feature_names, transformed):
    # for each topic in the transformed data (5 of them) calculate the sum of topic membership

    # row = lab, column = topic contribuion
    labSums = np.zeros(shape=(4, 5), dtype=np.float32)

    for abstractIndex in range(0, len(abstracts)):
        labs = abstractLabs[abstractIndex]
        for lab in labs:
            labIndex = list(mit_labs).index(lab)
            for topicIndex in range(0, 5):
                labSums[labIndex][topicIndex] = labSums[labIndex][topicIndex] + \
                    transformed[abstractIndex][topicIndex]

    labMemberships = np.zeros(shape=(4, 5), dtype=np.float32)
    for labIndex in range(0, 4):
        for topicIndex in range(0, 5):
            labMemberships[labIndex][topicIndex] = labSums[labIndex][topicIndex] / \
                np.sum(labSums[labIndex, :])

    plt.subplot(2, 9, (10, 19), title='Topics per MIT Lab')

    # we must plot each topic membership as a bar with a left offset to stack the bars
    #   for ex see https://matplotlib.org/3.3.3/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html
    # for topicIndex in range(0, 5):
    #     plt.barh(mit_labs, )

    leftOffsets = np.zeros(4, dtype=np.float32)
    for topicIndex in range(0, 5):
        xValues = labMemberships[:, topicIndex]
        plt.barh(list(mit_labs), xValues, left=leftOffsets,
                 label='Topic ' + str(topicIndex + 1))
        for i in range(0, len(xValues)):
            leftOffsets[i] = leftOffsets[i] + xValues[i]


    
    plt.legend(ncol=len(mit_labs), bbox_to_anchor=(1, 1.1),
              loc='right', fontsize='small')


def processAbstracts():
    tf_vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=n_features,
        stop_words='english'
    )
    tf = tf_vectorizer.fit_transform(abstracts)

    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=5,
        learning_method='online',
        learning_offset=50.,
        random_state=0
    )
    transformed = lda.fit_transform(tf)

    tf_feature_names = tf_vectorizer.get_feature_names()
    plotTopWords(lda, tf_feature_names, n_top_words)
    plotTopicBreakdown(lda, tf_feature_names, transformed)


# read all faculty member names
facultyDatas = json.loads(open('data/faculty', 'r').read())
processedCount = 0
for faculty in facultyDatas:
    processAbstractsForFacultyMember(faculty)
    processedCount = processedCount + 1

# prep the figure/plot
plt.figure(figsize=(12, 10))
plt.tight_layout()
plt.suptitle('Topics in LDA model', fontsize=16)

# process the abstracts for LDA and plot
processAbstracts()

# show the plot
plt.show()

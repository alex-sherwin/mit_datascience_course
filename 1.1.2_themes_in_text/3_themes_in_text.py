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

# all MIT labs
mit_labs = set(['CSAIL', 'LIDS', 'MTL', 'RLE'])

# list to hold all abstracts
abstracts = []

# list of labs per abstract (may be > 1 lab per abstract, this is determined by how many labs a faculty belongs to)
abstractLabs = []

# LDA algorithm tweakable params
n_features = 1000
n_components = 5
n_top_words = 10


# for each faculty member, open their articles JSON file, append it to the abstracts and track the labs for that abstract
def processAbstractsForFacultyMember(facultyMember):
    facultyName = facultyMember['name']
    labs = facultyMember['labs']
    articleData = json.loads(
        open('data/articles/' + facultyName, 'r').read()
    )
    for article in articleData['articles']:
        abstracts.append(article['abstract'])
        abstractLabs.append(labs)

# using LDA model output, plot the 5 topics top words


def plotTopWords(ldaModel, feature_names, n_top_words):
    for topic_idx, topic in enumerate(ldaModel.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        subplot_offset = topic_idx * 2

        plt.subplot(2, 9, subplot_offset + 1,
                    title='Topic ' + str(topic_idx+1))
        plt.barh(top_features, weights)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.invert_yaxis()
        plt.tick_params(axis='both', which='major', labelsize=10, pad=10)


# using LDA model transformed data, figure out each of the 5 topic memberships
# from 0-100% for each MIT lab and plot it in a stacked bar graph
def plotTopicBreakdown(ldaModel, feature_names, transformed):
    # for each topic in the transformed data (5 of them) calculate the sum of topic membership

    # 2-d vector to hold sum of topic contribution per lab
    # row = lab, column = topic contribution
    labSums = np.zeros(shape=(4, 5), dtype=np.float32)

    # for every abstract
    for abstractIndex in range(0, len(abstracts)):
        # check what labs this abstract belongs to
        labs = abstractLabs[abstractIndex]
        # for each lab this abstract belongs to
        for lab in labs:
            # get the index of this lab
            labIndex = list(mit_labs).index(lab)
            # for each topic
            for topicIndex in range(0, 5):
                # add to the running total sum of this lab's topic contribution
                labSums[labIndex][topicIndex] = labSums[labIndex][topicIndex] + \
                    transformed[abstractIndex][topicIndex]

    # 2-d vector to hold 0-100% membership values per lab/topic
    # row = lab, column = topic % membership
    labMemberships = np.zeros(shape=(4, 5), dtype=np.float32)

    # for every lab
    for labIndex in range(0, 4):
        # for every topic
        for topicIndex in range(0, 5):
            # calculate the % membership
            labMemberships[labIndex][topicIndex] = labSums[labIndex][topicIndex] / \
                np.sum(labSums[labIndex, :])

    # create a single subplot for our stacked bargraph
    plt.subplot(2, 9, (10, 19), title='Topics per MIT Lab')

    # we must plot each topic membership as a bar with a left offset to stack the bars
    #   for ex see https://matplotlib.org/3.3.3/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html
    # for topicIndex in range(0, 5):
    #     plt.barh(mit_labs, )

    # vector to hold left offsets for the stacked bar values
    leftOffsets = np.zeros(4, dtype=np.float32)

    # for each topic, re-plot a bar graph
    for topicIndex in range(0, 5):
        # slice out the membership values for this topic
        xValues = labMemberships[:, topicIndex]
        # plot the membership values where y=labs, x=membership % and left offsets are
        # accumulated between successicely stacked bars
        plt.barh(list(mit_labs), xValues, left=leftOffsets,
                 label='Topic ' + str(topicIndex + 1))

        # accumulate left offsets from current x values
        for i in range(0, len(xValues)):
            leftOffsets[i] = leftOffsets[i] + xValues[i]

    # show a topic legend
    plt.legend(ncol=len(mit_labs), bbox_to_anchor=(1, 1.1),
               loc='right', fontsize='small')

# process all abstracts and apply LDA
def processAbstractsAndPlot():

    # count/vectorize all the abstract texts
    tf_vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=n_features,
        stop_words='english'
    )
    tf = tf_vectorizer.fit_transform(abstracts)

    # apply LDA to the counted/vectorized abstract output
    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=5,
        learning_method='online',
        learning_offset=50.,
        random_state=0
    )
    transformed = lda.fit_transform(tf)

    # extract our feature names
    tf_feature_names = tf_vectorizer.get_feature_names()

    # plot the top words per topic
    plotTopWords(lda, tf_feature_names, n_top_words)

    # plot the per lab membership of topics
    plotTopicBreakdown(lda, tf_feature_names, transformed)


# read all faculty member names and process their abstracts
facultyDatas = json.loads(open('data/faculty', 'r').read())
for faculty in facultyDatas:
    processAbstractsForFacultyMember(faculty)

# prep the figure/plot
plt.figure(figsize=(12, 10))
plt.tight_layout()
plt.suptitle('Topics in LDA model', fontsize=16)

# process the abstracts for LDA and plot everything
processAbstractsAndPlot()

# show the plot
plt.show()

#!/usr/bin/env python3

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import Counter
from itertools import product

# read the data line-by-line from file (each line is already a 300 character chunk of DNA sequence)
inFile = open('genetic_codes_300_per_line', 'r')
# inFile = open('testline', 'r')
dnaLines = inFile.readlines()

def processLineForWords(dnaLine, currentDataPoint, wordSize, featureNumberMap):
    # break up the dnaLine into words of size wordSize
    wordsInLine = [
        dnaLine[i:i + wordSize]
        for i in range(0, len(dnaLine), wordSize)
    ]
    # use Counter to count the # of word occurrences (feature counts)
    counted = Counter(wordsInLine)

    # for each counted feature, find out what it's index is in the data point and set the count
    for word, count in counted.items():
        # lookup the index of the feature to use for the data point vector
        featureIndex = featureNumberMap[word]
        # set the feature count in the data point vector
        currentDataPoint[featureIndex] = count


# prep the plot visualization
plt.figure(figsize=(10, 10))

# for word sizes 1-4
for wordSize in range(1, 5):

    # generates all possible feature strings (dna sequences of a, c, g & t)
    features = list(
        map(lambda x: ''.join(x), list(product('acgt', repeat=wordSize)))
    )

    # take all possible features and create a map of { [feature]: [index] }
    # this is to have a consistent index per feature in the data point
    featureNumberMap = {}
    featureNumber = 0
    for feature in features:
        featureNumberMap[feature] = featureNumber
        featureNumber = featureNumber + 1

    # create our 2-dimensional array that will have:
    #   1018 rows [# data points]
    #   N feature columns (1 column per feature)
    #
    # use numpy.zeros() to create the two dimensional array seeded with 0.0 (floating point) values
    featurizedDataPoints = np.zeros(
        shape=(len(dnaLines), pow(4, wordSize)),
        dtype=np.float32
    )
    # iterate for every data point index
    for index in range(0, len(featurizedDataPoints)):
        # process the 300 character DNA sequence line for words and count them
        processLineForWords(
            dnaLines[index].strip(),  # the line of dna
            featurizedDataPoints[index],  # the data point vector of features
            wordSize,  # current word size (1-4)
            featureNumberMap  # lookup map of feature -> feature vector index
        )

    # ok great, featurizedDataPoints now contains a fully featurized data set of
    # 1018 rows of [featureN] features, where each row is a vector of [featureN] values
    # which are a count of how many times each possible feature occurs in the data point

    # normalize the data points
    normalizedDataPoints = StandardScaler().fit_transform(featurizedDataPoints)

    # we need to run PCA on this normalized [featureN]-dimensional data
    # to bring it down to 2 dimensional data to plot
    pca = PCA(n_components=2).fit_transform(normalizedDataPoints)

    kMeansResult = None

    # we happen to know we only want to run K-Means with n_clusters=7 on word size 3
    if wordSize == 3:
        # run K-Means on the normalized data to obtain clustering (to be used as plot coloring data)
        kMeansResult = KMeans(
            n_clusters=7,
            n_init=10,
        ).fit_predict(pca)

    # setup a subplot (graph)
    plt.subplot(2, 2, wordSize)

    # plot PCA reduced data
    plt.xlim((-9, 9))
    plt.ylim((-9, 9))
    plt.scatter(
        c=kMeansResult,  # use K-means result for coloring
        x=pca[:, 0],  # slice out x values
        y=pca[:, 1],  # slice out y values
        marker=".",  # draw a dot
        s=3  # point size
    )
    plt.title("DNA Word Size " + str(wordSize))
    plt.tight_layout()

# show the whole plot
plt.show()

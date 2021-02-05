#!/usr/bin/env python3

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# genetic code possible characters are a, c, g & t (note our data set is in lowercase)
letters = ['a', 'c', 'g', 't']
featureNumberMap = {}

featureNumber = 0

# iterate over the characters 3x to generate all possible 3-letter word combinations
# save these into featureNumberMap as a lookup map, where each possible word generates
# an index starting at 0 (which will generate 64 total indexes, from 0 to 63)
for firstLetter in letters:
    for secondLetter in letters:
        for thirdLetter in letters:
            featureName = firstLetter + secondLetter + thirdLetter
            featureNumberMap[featureName] = featureNumber
            featureNumber = featureNumber + 1

# read the data line-by-line from file (each line is already a 300 character chunk of DNA sequence)
inFile = open('genetic_codes_300_per_line', 'r')
dnaLines = inFile.readlines()

# create our 2-dimensional array that will have:
#   1018 rows [data points] (1 row per 300 character long DNA sequence)
#   64 columns (1 column per feature, which is each possible 3-letter word)
#
# use numpy.zeros() to create the two dimensional array seeded with 0.0 (floating point) values
featurizedDataPoints = np.zeros(shape=(len(dnaLines), 64), dtype=np.float32)


def processLineForWords(dnaLine, dataPointIndex):
    """
      for a given DNA sequence line, iterate over all possible combinations (64, each of our features)
      and count both it's occurrence, and it's occurrence in the reverse of the line
    """
    # get a pointer to the current data point (row of 64 features for the current DNA line)
    currentDataPoint = featurizedDataPoints[dataPointIndex]
    # get the reverse of the DNA line
    dnaLineReverse = dnaLine[::-1]

    # for each possible DNA 3-letter word (key = DNA word, value = data point feature index)
    for key, value in featureNumberMap.items():
        # count the occurrences of the DNA 3-letter word in both the DNA line and the reverse of the DNA line
        count = dnaLine.count(key) + dnaLineReverse.count(key)
        # record the occurrence count in the appropriate feature of the current data point
        currentDataPoint[value] = count


# for every data point index (300 character DNA sequence)
for dataPointIndex in range(0, len(featurizedDataPoints)):
    # process the 300 character DNA sequence line for words and count them
    processLineForWords(dnaLines[dataPointIndex], dataPointIndex)

# ok great, featurizedDataPoints now contains a fully featurized data set of
# 1018 rows of 64 features, where each row is an array of 64 values which are a count
# of how many times each possible word occurs in the data point

# we need to run PCA on this 64 dimensional data to bring it down to 2 dimensional data to plot

# normalize the data points
normalizedDataPoints = StandardScaler().fit_transform(featurizedDataPoints)
pca = PCA(n_components=2).fit_transform(normalizedDataPoints)

plt.subplot(1, 1, 1)

# plot PCA reduced data
plt.scatter(
    x=pca[:, 0],
    y=pca[:, 1],
    marker="."
)
plt.title("DNA Words PCA Result")

plt.show()

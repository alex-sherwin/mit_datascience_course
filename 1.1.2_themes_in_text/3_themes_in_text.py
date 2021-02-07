#!/usr/bin/env python3

import json
from nltk import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


n_features = 1000
n_components = 5
n_top_words = 10



def processFacultyName(name):
    articleData = json.loads(open('data/articles/' + name, 'r').read())

    abstracts = []
    for article in articleData['articles']:
        abstracts.append(article['abstract'])

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
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    plot_top_words(lda, tf_feature_names, n_top_words, 'Topics in LDA model')



    # for article in articleData['articles']:
    #     tokenized = word_tokenize(article['abstract'])
    #     counts = Counter(tokenized)
    #     print(counts)
# read all faculty member names
facultyDatas = json.loads(open('data/faculty', 'r').read())


# for each faculty member, process (perform an arXiv search, scrape results, save as JSON to a local file)
# for facultyData in facultyDatas:
#     name = facultyData['name']


processFacultyName(facultyDatas[1]['name'])

# for nameWithNewline in facultyNames:
#     name = nameWithNewline.strip()
#     articleData = json.loads(open('data/articles/' + name, 'r').read())

#     for article in articleData.articles:
#         nltk.tokenize(article.abstract)

# X, _ = make_multilabel_classification(random_state=0)
# lda = LatentDirichletAllocation(n_components=5, random_state=0)
# lda.fit(X)
# transformed = lda.transform(X[-2:])
# print("hi")

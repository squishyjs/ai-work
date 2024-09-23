import nltk
import csv
import nltk
import itertools
import re
from collections import Counter
# import markovify
import math
import os

from scipy.spatial.distance import cosine
from nltk.corpus import wordnet
import numpy as np
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings("ignore")

# ______________________________________________________________________________
# cfg
class CFG:
    def __init__(self, from_string):
        self.grammar = nltk.CFG.fromstring(from_string)
        self.parser = nltk.ChartParser(self.grammar)

    def change_grammar(self, from_string):
        self.grammar = nltk.CFG.fromstring(from_string)

    def draw_tree(self):
        self.sentence = input("Sentence: ").split()
        try:
            for tree in self.parser.parse(self.sentence):
                tree.pretty_print()
                tree.draw()
        except ValueError:
            print("No parse tree possible.")
# ______________________________________________________________________________
# ngrams

class Ngrams:
    def __init__(self, directory, n = 1):
        self.directory = directory
        self.n = n

    def load_data_ngrams(self):
        self.contents = []

        # Read all files and extract words
        for filename in os.listdir(self.directory):
            with open(os.path.join(self.directory, filename), encoding = 'UTF-8') as f:
                self.contents.extend([
                    word.lower() for word in
                    nltk.word_tokenize(f.read())
                    if any(c.isalpha() for c in word)
                ])
        return self.contents

    def top_term_frequencies(self, n = 10):
        """Calculate top term frequencies for a corpus of documents."""

        corpus = self.load_data_ngrams()

        # Compute n-grams
        ngrams = Counter(nltk.ngrams(corpus, self.n))

        # Print most common n-grams
        for ngram, freq in ngrams.most_common(n):
            print(f"{freq}: {ngram}")

# ______________________________________________________________________________
# markov
class Markov:
    def __init__(self, directory):
        self.directory = directory
        with open(self.directory) as f:
            text = f.read()
        self.text_model = markovify.Text(text)

    def creat_sentence(self, n = 5):
        for i in range(n):
            print(self.text_model.make_sentence())
            print()
# ______________________________________________________________________________
# sentiment

class Sentiment:
    def __init__(self, directory, file_name):
        self.directory = directory
        self.positives = []
        self.negatives = []
        self.words = set()
        self.file_name = file_name

    def create_set_of_words(self):
        self.positives, self.negatives = self.load_data()
        for document in self.positives:
            self.words.update(document)
        for document in self.negatives:
            self.words.update(document)
        return (self.positives, self.negatives)

    def extract_words(self, document):
        return set(
            word.lower() for word in nltk.word_tokenize(document)
            if any(c.isalpha() for c in word)
        )


    def load_data(self):
        self.result = []
        for filename in self.file_name:
            with open(os.path.join(self.directory, filename), encoding = 'UTF-8') as f:
                self.result.append([
                    self.extract_words(line)
                    for line in f.read().splitlines()
                ])
        return self.result


    def generate_features(self, documents, label):
        features = []
        for document in documents:
            features.append(({
                word: (word in document)
                for word in self.words
            }, label))
        return features


    def classify(self, classifier, document):
        document_words = self.extract_words(document)
        features = {
            word: (word in document_words)
            for word in self.words
        }
        return classifier.prob_classify(features)


# ______________________________________________________________________________
# term_frequency
class TermFrequency:
    def __init__(self, directory):
        self.directory = directory

    def term_frequency(self, corpus, num_words):
            # Get all words in corpus
        print("Extracting words from corpus...")
        words = set()
        for filename in corpus:
            words.update(corpus[filename])

        # Calculate IDFs
        print("Calculating inverse document frequencies...")
        idfs = dict()
        for word in words:
            f = sum(word in corpus[filename] for filename in corpus)
            idf = math.log(len(corpus) / f)
            idfs[word] = idf

        # Calculate TF-IDFs
        print("Calculating term frequencies...")
        tfidfs = dict()
        for filename in corpus:
            tfidfs[filename] = []
            for word in corpus[filename]:
                tf = corpus[filename][word]
                tfidfs[filename].append((word, tf * idfs[word]))

        # Sort and get TF-IDFs of top num_words in each file
        print("Computing top terms...")
        for filename in corpus:
            tfidfs[filename].sort(key=lambda tfidf: tfidf[1], reverse=True)
            tfidfs[filename] = tfidfs[filename][:num_words]

        # Print results
        print()
        for filename in corpus:
            print(filename)
            for term, score in tfidfs[filename]:
                print(f"    {term}: {score:.4f}")

    def load_data(self):
        files = dict()
        for filename in os.listdir(self.directory):
            with open(os.path.join(self.directory, filename), encoding = 'UTF-8') as f:

                # Extract words
                contents = [
                    word.lower() for word in
                    nltk.word_tokenize(f.read())
                    if word.isalpha()
                ]

                # Count frequencies
                frequencies = dict()
                for word in contents:
                    if word not in frequencies:
                        frequencies[word] = 1
                    else:
                        frequencies[word] += 1
                files[filename] = frequencies

        return files

    def load_function(self):

        with open("function_words.txt") as f:
            function_words = set(f.read().splitlines())

        files = dict()
        for filename in os.listdir(self.directory):
            with open(os.path.join(self.directory, filename), encoding = 'UTF-8') as f:

                # Extract words
                contents = [
                    word.lower() for word in
                    nltk.word_tokenize(f.read())
                    if word.isalpha()
                ]

                # Count frequencies
                frequencies = dict()
                for word in contents:

                    if word in function_words:
                        continue
                    elif word not in frequencies:
                        frequencies[word] = 1
                    else:
                        frequencies[word] += 1
                files[filename] = frequencies

        return files

# ______________________________________________________________________________
# word_net

def print_word_net(synsets):
    for synset in synsets:
        print()
        print(f"{synset.name()}: {synset.definition()}")
        for hypernym in synset.hypernyms():
            print(f"  {hypernym.name()}")


# ______________________________________________________________________________
# search
class Search:
    def __init__(self, data, directory):
        self.data = data
        self.directory = directory

    def load(self):
        with open(self.data) as f:
            examples = list(csv.reader(f))
        corpus = ""
        for filename in os.listdir(self.directory):
            with open(os.path.join(self.directory, filename), encoding='UTF-8') as f:
                corpus += f.read().replace("\n", " ")
        return examples, corpus


    def find_templates(self, examples, corpus):
        templates = []
        for a, b in examples:
            templates.extend(self.match_query(a, b, True, corpus))
            templates.extend(self.match_query(b, a, False, corpus))

        # Find common middles
        middles = dict()
        for template in templates:
            middle = template["middle"]
            order = template["order"]
            if (middle, order) in middles:
                middles[middle, order].append(template)
            else:
                middles[middle, order] = [template]

        # Filter middles to only those used multiple times
        middles = {
            middle: middles[middle]
            for middle in middles
            if len(middles[middle]) > 1
        }

        # Look for common prefixes and suffixes
        results = []
        for middle in middles:
            found = set()
            for t1, t2 in itertools.combinations(middles[middle], 2):
                prefix = self.common_suffix(t1["prefix"], t2["prefix"])
                suffix = self.common_prefix(t1["suffix"], t2["suffix"])
                if (prefix, suffix) not in found:
                    if (not len(prefix) or not len(suffix)
                       or not prefix.strip() or not suffix.strip()):
                            continue
                    found.add((prefix, suffix))
                    results.append({
                        "order": middle[1],
                        "prefix": prefix,
                        "middle": middle[0],
                        "suffix": suffix
                    })
        results = self.filter_templates(results)
        return results


    def filter_templates(self, templates):
        return sorted(
            templates,
            key=lambda t: len(t["prefix"]) + len(t["suffix"]),
            reverse=True
        )


    def extract_from_templates(self, templates, corpus):
        results = set()
        for template in templates:
            results.update(self.match_template(template, corpus))
        return results


    def match_query(self, q1, q2, order, corpus):
        q1 = re.escape(q1)
        q2 = re.escape(q2)
        regex = f"(.{{0,10}}){q1}((?:(?!{q1}).)*?){q2}(.{{0,10}})"
        results = re.findall(regex, corpus)
        return [
            {
                "order": order,
                "prefix": result[0],
                "middle": result[1],
                "suffix": result[2]
            }
            for result in results
        ]


    def match_template(self, template, corpus):
        prefix = re.escape(template["prefix"])
        middle = re.escape(template["middle"])
        suffix = re.escape(template["suffix"])
        regex = f"{prefix}((?:(?!{prefix}).){{0,40}}?){middle}(.{{0,40}}?){suffix}"
        results = re.findall(regex, corpus)
        if template["order"]:
            return results
        else:
            return [(b, a) for (a, b) in results]


    def common_prefix(self, *s):
        # https://rosettacode.org/wiki/Longest_common_prefix#Python
        return "".join(
            ch[0] for ch in itertools.takewhile(
                lambda x: min(x) == max(x), zip(*s)
            )
        )


    def common_suffix(self, *s):
        s = [x[::-1] for x in list(s)]
        return self.common_prefix(*s)[::-1]

# ______________________________________________________________________________
# vectors
class Vectors:
    def __init__(self, directory):
        self.directory = directory
        self.words = dict()
        with open((self.directory), encoding = 'UTF-8') as f:
            for i in range(50000):
                row = next(f).split()
                word = row[0]
                vector = np.array([float(x) for x in row[1:]])
                self.words[word] = vector

    def distance(self, w1, w2):
        return cosine(w1, w2)

    def closest_words(self, embedding):
        distances = {
            w: self.distance(embedding, self.words[w])
            for w in self.words
        }
        return sorted(distances, key=lambda w: distances[w])[:10]


    def closest_word(self, embedding):
        return self.closest_words(embedding)[0]


def find_features(document, word_features):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

def extract_features(document, word_features):
    feature = []
    for rev, category in document:
        feature.append((find_features(rev, word_features), category))
    return feature

def char_dices(chars):
    return (dict((c, i) for i, c in enumerate(chars)), dict((i, c) for i, c in enumerate(chars)))

def create_train_label_data(text, char_num,  char_indices, indices_char, maxlen, step):
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    train_data = np.zeros((len(sentences), maxlen, char_num), dtype=np.bool)
    label_data = np.zeros((len(sentences), char_num), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            train_data[i, t, char_indices[char]] = 1
        label_data[i, char_indices[next_chars[i]]] = 1
    return (train_data, label_data)

def word_feats(words):
    return dict([(word, True) for word in words.split()])
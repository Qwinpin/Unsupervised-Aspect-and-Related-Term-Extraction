import json
import xml.etree.ElementTree as ET

import numpy as np

from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


def parse_xml(path='../data/ABSA16_Restaurants_Train_SB1_v2.xml'):
    root = ET.parse('path').getroot()
    return root


def get_text_from_xml(xml):
    texts = []
    texts_merged = []

    labels = []
    labels_merged = []

    for review in xml.findall('Review'):
        _texts = ''
        _labels_all = {}

        for sentence in review.find('sentences').findall('sentence'):
            text = sentence.find('text').text
            texts.append(text)
            _texts += ' {}'.format(text)


            op = sentence.find('Opinions')
            if op is not None:
                _labels = {}
                for label in sentence.find('Opinions').findall('Opinion'):
                    target_term = label.get('target')
                    target_aspect = label.get('category')
                    _labels[target_aspect] = target_term

                labels.append(_labels)
                _labels_all.update(_labels)
            else:
                labels.append({})

        texts_merged.append(_texts)
        labels_merged.append(_labels_all)

    return texts, texts_merged, labels, labels_merged


def process_text(text, vocabulary_reverse, weights_matrix, stopWords):
    tokens = word_tokenize(text.lower())
    tokens = [i for i in tokens if i not in stopWords]

    idx = [vocabulary_reverse.get(i, 1) for i in tokens]
    emb = [weights_matrix[i] for i in idx]
    return idx, emb


def process_restraunt(path='../data/ABSA16_Restaurants_Train_SB1_v2.xml', model_path='../data/w2v.txt', save=True):
    root = parse_xml(path)
    model = KeyedVectors.load_word2vec_format("gensim_glove_w2v_vectors.txt", binary=False)

    vocabulary = {0:'<pad>', 1:'<unk>', 2:'<num>'}
    vocabulary_reverse = {'<pad>':0, '<unk>':1, '<num>':2}
    model_vocab = list(model.wv.vocab.keys())
    stopWords = set(stopwords.words('english'))

    texts, texts_merged, labels, labels_merged = get_text_from_xml(root)

    idx = 3
    for text in texts:
        tokens = word_tokenize(text.lower())
        tokens = [i for i in tokens if i not in stopWords]
        for i, token in enumerate(tokens):
            if token in model_vocab:
                vocabulary[idx] = token
                vocabulary_reverse[token] = idx
                idx += 1
            else:
                _idx = 2 if token.isdigit() else 1
                vocabulary[_idx] = token
                vocabulary_reverse[token] = _idx

    matrix_len = len(vocabulary)
    weights_matrix = np.zeros((matrix_len, 200))

    for word, i in vocabulary_reverse.items():
        if word in ['<num>', '<unk>', '<pad>']:
            weights_matrix[i] = np.zeros(200) + 1e-10
        else:
            try:
                weights_matrix[i] = model.wv.word_vec(word)
            except:
                weights_matrix[i] = np.zeros(200) + 1e-10

    weights_matrix_norm = weights_matrix / np.linalg.norm(weights_matrix, axis=-1, keepdims=True)

    texts_tokens = [process_text(text, vocabulary_reverse, weights_matrix_norm, stopWords)[0] for text in texts]
    texts_merged_tokens = [process_text(text, vocabulary_reverse, weights_matrix_norm, stopWords)[0] for text in texts_merged]
    indexes = list(range(len(texts)))

    np.random.seed(42)
    np.random.shuffle(indexes)
    indexes_train = indexes[:len(indexes) // 100 * 85]
    indexes_test = indexes[len(indexes) // 100 * 85:]

    data = {}
    data['texts'] = texts
    data['labels'] = labels
    data['texts_merged'] = texts_merged
    data['labels_merged'] = labels_merged
    data['tokens'] = texts_tokens
    data['tokens_merged'] = texts_merged_tokens
    data['train_indexes'] = indexes_train
    data['test_indexes'] = indexes_test
    data['vocabulary'] = vocabulary
    data['vocabulary_reverse'] = vocabulary_reverse
    data['weights_matrix_norm'] = weights_matrix_norm

    if save:
        with open('ABSA16_Restaurants_Train_SB1_v2_processed.json', 'w') as f:
            json.dump(data, f)

    return data

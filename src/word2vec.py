from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec


def vectorizer(
    corpus: List[List[str]], model: Word2Vec, num_features: int = 100
) -> np.ndarray:
    """
    This function takes a list of tokenized text documents (corpus) and a pre-trained
    Word2Vec model as input, and returns a matrix where each row represents the
    vectorized form of a document.

    Args:
        corpus : list
            A list of text documents that needs to be vectorized.

        model : Word2Vec
            A pre-trained Word2Vec model that will be used to vectorize the corpus.

        num_features : int
            The size of the vector representation of each word. Default is 100.

    Returns:
        corpus_vectors : numpy.ndarray
            A 2D numpy array where each row represents the vectorized form of a
            document in the corpus.
    """
    # TODO
    # loop through every row
    averaged_vectors = []
    not_found = []
    for row in corpus:
        total_tokens = 0
        vct_sum = np.zeros(num_features)
        ## loop through every tokenized word, get the vector and sum
        for token in row:
            try:
                vct_sum = vct_sum + model.wv.get_vector(token)
                total_tokens += 1
            except Exception:
                try:
                    vct_sum = vct_sum + model[token]
                    total_tokens += 1
                except:
                    not_found.append(token)
        # get the average of all of those vectors in the review, put in the array
        averaged_vectors.append(vct_sum / total_tokens)

    return np.array(averaged_vectors)
# 1

def w2v_information(model: Word2Vec):
    print("w2vec information:")
    print("-" * 30)
    print("Corpus count: {}".format(model.corpus_count))
    print("Trained with: {} words".format(model.corpus_total_words))
    print("Vocabulary found: {}".format(len(model.wv)))
    print("-" * 30)
    print("Example word: {}".format(model.wv.index_to_key[0]))
    print(model.wv[0])

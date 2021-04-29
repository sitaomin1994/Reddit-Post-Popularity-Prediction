import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def evaluation(y_true, y_pred):
    rho, _ = stats.spearmanr(y_true, y_pred)
    print("=====================================")
    print("Result:")
    print("Spearman Rho: {}".format(rho))
    print("MAE: {:.4f}".format(mean_absolute_error(y_true, y_pred)))
    print("RMSE: {:.4f}".format(mean_squared_error(y_true, y_pred)))

def mean_sentence_embedding(sentences, model, dim=300):
    sentences_vec = []
    for sentence in sentences:
        token_vecs = []
        for token in sentence:
            if token not in model.wv.index2word:
                token_vecs.append(np.random.rand(dim))
            else:
                token_vecs.append(model.wv[token])
        vectors = np.stack(token_vecs, axis=0)
        sentences_vec.append(np.mean(vectors, axis=0))

    return np.stack(sentences_vec, axis=0)



def split_train_test(X, y, ravel=False):
    if ravel == True:
        y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=21)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=21)

    print('The size of X_train is:', X_train.shape, '\nThe size of y_train is:', y_train.shape,
          '\nThe size of X_val is:', X_val.shape, '\nThe size of y_val is:', y_val.shape,
          '\nThe size of X_test is:', X_test.shape, '\nThe size of y_test is:', y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test
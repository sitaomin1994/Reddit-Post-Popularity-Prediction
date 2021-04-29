from NNRegressor import NNRegressor
from Dataloader import encode_titles, Eluvio, split_train_test
from torch.utils.data import DataLoader
from model import LSTM_FUSI
import torch
import pandas as pd
import random
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from Dataloader import load_pretained_embedding, load_pretained_embedding_bert


def evaluation(y_true, y_pred):
    rho, _ = stats.spearmanr(y_true, y_pred)
    print("=====================================")
    print("Result:")
    print("Spearman Rho: {}".format(rho))
    print("MAE: {:.4f}".format(mean_absolute_error(y_true, y_pred)))
    print("RMSE: {:.4f}".format(mean_squared_error(y_true, y_pred)))


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    data = pd.read_csv('./Data/data.csv')
    data['month'] = data['month'] - 1
    data['day'] = data['day'] - 1
    data['dayofyear'] = data['dayofyear'] -1
    data['week'] = data['week'] -1

    # some parameters
    dense_col = ['over_18', 'title_len', 'title_num_char']
    num_dense = len(dense_col)
    labelencoder = LabelEncoder()
    data['author_code'] = labelencoder.fit_transform(data['author'])
    data['year_code'] = labelencoder.fit_transform(data['year'])
    num_author = data['author_code'].nunique()
    num_year = data['year_code'].nunique()

    # generate pad sequences of titles
    embed_dim = 300
    titles, vocabulary, vocabulary_inv = encode_titles(data['title_cleaned'])
    weights = load_pretained_embedding(vocabulary_inv, embed_dim=embed_dim)
    #weights = np.load('./Embed/bert_emb768.npy')
    vocab_size = len(vocabulary_inv.keys())
    print(weights.shape)

    # learning params
    batch_size = 256
    lr = 0.001

    data_train, data_val, data_test, titles_train, titles_val, titles_test = \
        split_train_test(data, data['up_votes'], titles)

    # Data
    train_data = Eluvio(data_train, titles_train, dense_col=dense_col)
    validation_data = Eluvio(data_val, titles_val, dense_col=dense_col)
    test_data = Eluvio(data_test, titles_test, dense_col=dense_col)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = LSTM_FUSI.LSTM_FUSI(vocab_size=vocab_size,
                                author_size=num_author,
                                batch_size=batch_size,
                                n_dense_features=num_dense,
                                num_year=num_year,
                                weights=weights,
                                embedding_dim=embed_dim)

    print(torch.cuda.is_available())
    model = model.cuda()

    # Regressor
    feature_list = ['year', 'hour', 'month', 'day', 'weekday', 'n_day', 'n_week']
    reg = NNRegressor(model, feature_list, lr=lr)
    reg.fit(train_dataloader, eval_set=validation_dataloader)

    print("=================================")
    y_pred = reg.predict(test_dataloader)
    y_test = data_test['log_up_votes'].values.reshape(-1, 1)
    evaluation(y_pred, y_test)

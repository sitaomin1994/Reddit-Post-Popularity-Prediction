import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertModel
import pandas as pd
from sister import word_embedders

class Eluvio(Dataset):

    def __init__(self, data, titles, dense_col=None, target_col='log_up_votes', padded=True,
                 transform=None):
        """
        Args:
            data: dataframe contains features and labels
            target_col : target columns name
        """
        self.data = data
        self.titles = titles
        self.titles = np.array(self.titles)
        #print(type(self.titles))
        #print(self.titles.shape)

        if dense_col is None:
            dense_col = ['over_18', 'title_len', 'title_num_char']

        # feature information
        self.author = self.data['author_code'].values.reshape(-1,1)
        self.year = self.data['year_code'].values.reshape(-1,1)
        self.month = self.data['month'].values.reshape(-1,1)
        self.day = self.data['day'].values.reshape(-1,1)
        self.hour = self.data['hour'].values.reshape(-1,1)
        self.weekday = self.data['weekday'].values.reshape(-1,1)
        self.n_week = self.data['week'].values.reshape(-1,1)
        self.n_day = self.data['dayofyear'].values.reshape(-1,1)
        # self.quarter = data[[col for col in data if col.startswith('quarter')]].values
        # other dense features
        self.dense = data[dense_col].astype('float32').values

        # whether to pad sequence
        self.padded = padded

        # label
        self.label = data[target_col].values.reshape(-1,1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sentence = self.titles[idx]
        if self.padded:
            length_arr = np.where(sentence == 0)[0]
            if length_arr.shape[0] == 0:
                length = sentence.shape[0]
            else:
                length = np.where(sentence == 0)[0][0]
        else:
            length = sentence.shape[0]

        sample = {'title': self.titles[idx],
                  'length': length,
                  'year': self.year[idx],
                  'author': self.author[idx],
                  'month': self.month[idx],
                  'day': self.day[idx],
                  'hour': self.hour[idx],
                  'weekday': self.weekday[idx],
                  'n_week': self.n_week[idx],
                  'n_day': self.n_day[idx],
                  # 'quarter': self.quarter[idx],
                  'dense': self.dense[idx],
                  'label': self.label[idx]}

        return sample

    def _oneHotEncode(self, X):
        X = X.values.reshape(-1, 1)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(X)
        return enc.transform(X).toarray()



class ToTensor(object):
    """Convert arrays in sample to Tensors."""

    def __call__(self, sample):
        result = {}

        for key, value in sample.items():
            result[key] = torch.from_numpy(value)

        return result


def split_train_test(X, y, titles):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=21)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=21)

    titles_train = titles[X_train.index]
    titles_validation = titles[X_val.index]
    titles_test = titles[X_test.index]

    return X_train, X_val, X_test, titles_train, titles_validation, titles_test

###########################################################################################
# Text handling
###########################################################################################

def encode_titles(titles, embed=True):
    t = Tokenizer(filters='', lower=False)
    t.fit_on_texts(titles)

    vocabulary = t.word_index
    vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
    vocabulary_inv[0] = '<PAD/>'
    vocabulary['<PAD/>'] = 0
    # convert title to sequences
    sequences = t.texts_to_sequences(titles)

    # pad sequences
    padded_seq = pad_sequences(sequences, padding="post", truncating="post")

    return np.array(padded_seq), vocabulary, vocabulary_inv


def bert_text_preparation(text, tokenizer):
    """Preparing the input for BERT

    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.

    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids

    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids


    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model

    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids

    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token

    """

    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings

def load_pretained_embedding(vocab_inv, embed_dim):
    vocab_size = len(vocab_inv.keys())
    weights = np.random.uniform(-0.25, 0.25, (vocab_size, embed_dim))
    word2vec = word_embedders.Word2VecEmbedding()
    for key, value in vocab_inv.items():
        if key == 0:
            weights[0] = np.zeros(embed_dim)
        else:
            weights[key] = word2vec.get_word_vector(value)
    return weights

def load_pretained_embedding_bert(vocab_inv, embed_dim):
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = len(vocab_inv.keys())
    weights = np.random.uniform(-0.25, 0.25, (vocab_size, embed_dim))
    for idx, (key, value) in enumerate(vocab_inv.items()):
        if idx % 500 == 0:
            print(idx)
        if key == 0:
            weights[0] = np.zeros(embed_dim)
        else:
            tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(value, tokenizer)
            list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
            weights[key] = list_token_embeddings[1]
    return weights

if __name__ == '__main__':

    data = pd.read_csv('./Data/data.csv')
    print(data.shape)

    texts = data['title_cleaned']

    padded_seq, vocabulary, vocabulary_inv = encode_titles(texts)

    emb = load_pretained_embedding_bert(vocabulary_inv, embed_dim=768)
    print(emb.shape)
    np.save('./Embed/bert_emb768.npy', emb)
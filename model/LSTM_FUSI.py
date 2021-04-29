import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM_FUSI(nn.Module):

    def __init__(self, vocab_size, author_size, num_year, n_dense_features, batch_size, weights=None,
                 embedding_dim=300, n_hidden=300, n_out=1, num_layers=1, bidirectional=True,
                 author_embed_dim=20, time_embed_dim=20):
        super(LSTM_FUSI, self).__init__()

        # LSTM_FUSI parameters
        self.batch_size = batch_size
        self.vocab_size = vocab_size  # vocabulary size
        self.embedding_dim = embedding_dim  # embedding dimension
        self.n_hidden = n_hidden  # hidden dimension
        self.n_out = n_out  # output dimension
        self.num_layers = num_layers  # number of LSTM layer
        self.bidirectional = bidirectional  # bidirectional LSTM
        self.author_embed_dim = author_embed_dim  # embedding for author information
        self.author_size = author_size
        self.time_embed_dim = time_embed_dim
        self.num_year = num_year
        self.n_dense_features = n_dense_features
        if weights is not None:
            self.weights = torch.from_numpy(weights).float().cuda()
            self.emb = nn.Embedding.from_pretrained(self.weights)
        else:
            self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)

        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # feature embedding
        self.author_emb = nn.Embedding(self.author_size, self.author_embed_dim)
        self.month_emb = nn.Embedding(12, self.time_embed_dim)
        self.year_emb = nn.Embedding(self.num_year, self.time_embed_dim)
        self.day_emb = nn.Embedding(31, self.time_embed_dim)
        self.hour_emb = nn.Embedding(24, self.time_embed_dim)
        self.weekday_emb = nn.Embedding(7, self.time_embed_dim)
        self.n_week_emb = nn.Embedding(54, self.time_embed_dim)
        self.n_day_emb = nn.Embedding(366, self.time_embed_dim)

        # LSTM
        self.lstm = nn.LSTM(self.embedding_dim, self.n_hidden, self.num_layers, bidirectional=self.bidirectional)

        # Fully connect layer
        self.all_dim = 2 * self.n_hidden + 2 * self.num_directions * self.n_hidden + self.n_dense_features\
                       + self.author_embed_dim + 7*self.time_embed_dim

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.all_dim, 512),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

        self.fc.apply(self.init_weights)

    def forward(self, data, feature_list):

        # =================================
        # sequence encoder BiLSTM
        # =================================
        # print(seq.size())
        input_sentences = data['title']  # input sentences dimension (batch_size, sequence_length)
        embed_sentences = self.emb(input_sentences)  # embedding => (batch_size, sequence_length, embed_dim)
        embed_sentences = embed_sentences.permute(1, 0, 2)  # inputs => (sequence_length, batch_size, embed_dim)

        batch_size = embed_sentences.size(1)
        if batch_size is not None:
            h0 = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size,
                                      self.n_hidden).cuda())  # Initial hidden state of the LSTM
            c0 = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size,
                                      self.n_hidden).cuda())  # Initial cell state of the LSTM
        else:
            h0 = Variable(torch.zeros(self.num_layers * self.num_directions, self.batch_size,
                                      self.n_hidden).cuda())  # Initial hidden state of the LSTM
            c0 = Variable(torch.zeros(self.num_layers * self.num_directions, self.batch_size,
                                      self.n_hidden).cuda())  # Initial cell state of the LSTM

        # embs = pack_padded_sequence(embs, lengths)
        embed_sentences = nn.Dropout(0.5)(embed_sentences)
        lstm_output, (h, c) = self.lstm(embed_sentences, (h0, c0))
        # lstm_out, lengths = pad_packed_sequence(lstm_out)
        avg_pool = F.adaptive_avg_pool1d(lstm_output.permute(1, 2, 0), 1).view(batch_size, -1)
        max_pool = F.adaptive_max_pool1d(lstm_output.permute(1, 2, 0), 1).view(batch_size, -1)

        #attn_output = self.attention_net(lstm_output.permute(1, 0, 2), h)   # output.size() = (batch_size, num_seq, hidden_size)
        #print(attn_output.shape)
        sentence_output = torch.cat([c[-1], c[-2], avg_pool, max_pool], dim=1)  # (batch_size, 3*num_directions*n_hidden)

        # additional feature embedding and stack
        feature_embs = [
                data['dense'],
                self.author_emb(data['author']).squeeze(1),
                self.year_emb(data['year']).squeeze(1),
                self.day_emb(data['day']).squeeze(1),
                self.month_emb(data['month']).squeeze(1),
                self.hour_emb(data['hour']).squeeze(1),
                self.weekday_emb(data['weekday']).squeeze(1),
                self.n_week_emb(data['n_week']).squeeze(1),
                self.n_day_emb(data['n_day']).squeeze(1)
            ]


        additional_feature_output = torch.cat(feature_embs, dim=1)

        # ==============================================
        # Fusing all features together
        # ==============================================
        #all_features = additional_feature_output
        all_features = torch.cat([sentence_output, additional_feature_output], dim=1)

        # Fully connected MLP
        out = self.fc(all_features)

        return out

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def attention_net(self, lstm_output, final_state):

        """
        use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """
        # final state => (num_direction*layers, batch size, hidden_size)
        # permuted lstm ouput => (batch size, seq_len, num_direction*hidden size)
        hidden = final_state.squeeze(0)
        print(final_state.shape)
        print(hidden.shape)
        print(hidden.unsqueeze(2).shape)
        print(lstm_output.shape)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

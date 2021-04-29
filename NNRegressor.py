import torch
import time
from torch import nn
import math
from sklearn.metrics import mean_squared_error
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F


class NNRegressor:

    def __init__(self, model, feature_list, verbose=1, lr=0.03,
                 optimizer=None, criterion=None, epochs=100, path=None):

        self.model = model
        self.verbose = verbose
        self.feature_list = feature_list
        self.epochs = 100 if epochs is None else epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr) if optimizer is None else optimizer
        self.criterion = nn.MSELoss() if criterion is None else criterion

        # track the best history
        self.best_epoch = 0
        self.path = 'best_model.pt' if path is None else path

    def fit(self, dataloader, eval_set=None, eval_metric=None, early_stopping=5):
        """
        Training process
        :param dataloader: training data from pytorch dataloader object
        :param eval_set: validation data from pytorch dataloader object
        :param eval_metric:  evaluation metrics default is sklearn.metrics.mean_square_error
        :param early_stopping: early stopping rounds
        :return:
        """

        self.best_epoch = 0
        log_interval = self.verbose
        start_time = time.time()
        best_result = math.inf
        es_track = 0  # early stopping tracker

        if eval_metric is None:
            eval_metric = mean_squared_error

        for epoch in range(1, self.epochs + 1):

            self.model.train()

            loss_list = []  # tracking the loss for each epoch

            for idx, data_dict in enumerate(dataloader):

                if torch.cuda.is_available():
                    for key, value in data_dict.items():
                        data_dict[key] = data_dict[key].cuda()

                y_train = data_dict['label']
                self.optimizer.zero_grad()
                y_pred = self.model(data_dict, self.feature_list)
                loss = self.criterion(y_pred.float(), y_train.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
                loss_list.append(loss.item())
            # evaluation
            if eval_set is not None:
                eval_result = self.evaluate(eval_set, eval_metric)
                if eval_result <= best_result:
                    best_result = eval_result
                    self.best_epoch = epoch
                    es_track = 0  # reset es_track
                    # save best model
                    torch.save(self.model.state_dict(), self.path)
                else:
                    es_track += 1  # update es_track
            else:
                eval_result = 0
                self.best_epoch = epoch

            # print information
            if epoch % log_interval == 0 and epoch > 0:
                elapsed = time.time() - start_time
                if eval_set is not None:
                    print(
                        '|{:3d} | loss: {:8.6f} | eval: {:8.6f} | best: {:8.6f} ({:3d}) | time: {:.2}'
                            .format(epoch, sum(loss_list) / len(loss_list), eval_result, best_result, self.best_epoch,
                                    elapsed))
                else:
                    print('| epoch {:3d} | loss: {:8.3f} | time: {:.2}'
                          .format(epoch, sum(loss_list) / len(loss_list), elapsed))

            # early stopping
            if es_track > early_stopping:
                break

    def evaluate(self, dataloader, eval_metric=None):
        """
        evaluate the model on validation data
        :param dataloader: pytorch Dataloader for validation data
        :param eval_metric: None default set to MSE
        :return: evaluation result
        """

        self.model.eval()
        eval_result = []

        if eval_metric is None:
            eval_metric = mean_squared_error

        with torch.no_grad():
            for idx, data_dict in enumerate(dataloader):

                if torch.cuda.is_available():
                    for key, value in data_dict.items():
                        data_dict[key] = data_dict[key].cuda()
                y_test = data_dict['label']
                y_pred = self.model(data_dict, self.feature_list)
                eval_result.append(eval_metric(y_pred.cpu().numpy(), y_test.cpu().numpy()))

        return sum(eval_result)/len(eval_result)

    def predict(self, dataloader):
        """
        Make prediction using text data
        :param dataloader: pytorch dataloader object
        :return: 1d numpy array of y_pred
        """
        self.model.load_state_dict(torch.load(self.path))
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for idx, data_dict in enumerate(dataloader):
                if torch.cuda.is_available():
                    for key, value in data_dict.items():
                        data_dict[key] = data_dict[key].cuda()
                y_pred.append(self.model(data_dict, self.feature_list).cpu().numpy())

        return np.concatenate(y_pred, axis=0)


if __name__ == '__main__':

    # features
    embedding1 = nn.Embedding(10, 5)
    embedding2 = nn.Embedding(10, 3)

    emb = [embedding1, embedding2]
    input1 = torch.LongTensor([[1], [4]])
    input2 = torch.LongTensor([[3], [4]])
    input3 = torch.LongTensor([[3, 10, 20], [4, 20, 20]])

    input = [input1, input2]
    output = []
    for i in range(len(input)):
        a = emb[i](input[i])
        a = a.squeeze(1)
        print(a)
        output.append(a)

    output.append(input3)

    output = torch.cat(output, dim=1)

    print(output.shape, output)
    print("=" * 100)

    # titles
    input3 = torch.LongTensor([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]])
    embedding3 = nn.Embedding(10, 8)
    output3 = embedding3(input3)
    output3 = output3.permute(1,0,2)
    print(output3.shape)

    lstm = nn.LSTM(8, 10, 1, bidirectional=True)
    batch_size = input3.size(0)
    print(batch_size)

    h = Variable(torch.zeros((2, 2, 10)))
    c = Variable(torch.zeros((2, 2, 10)))

    lstm_out, (h, c) = lstm(output3, (h, c))

    print(lstm_out, lstm_out.shape) #lstm out => (sequence len, batch size, hidden_dim*2)
    # (batch_size, hidden_dim*2, sequence_len)
    avg_pool = F.adaptive_avg_pool1d(lstm_out.permute(1, 2, 0), 1).view(batch_size, -1)
    max_pool = F.adaptive_max_pool1d(lstm_out.permute(1, 2, 0), 1).view(batch_size, -1)
    sentence_output = torch.cat([c[-1], avg_pool, max_pool], dim=1)  # (batch_size, 3*num_directions*n_hidden)

    print(sentence_output.shape)

    all_tog = torch.cat([output, sentence_output], dim = 1)

    linear = nn.Linear(61,1)

    out = linear(all_tog)

    print(out)

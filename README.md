## Reddit Post Popularity Prediction

The dataset contains 500,000 reddit news posts including information of post itme, author, over-18, and titles. The data is tabular and the features involved should be self-explanatory.

### Prediction task: predict log upvotes of reddit post
- Target: log upvotes of reddit
- Evaluation metrics: RMSE, MAE, Spearman Rho

### Instruction

- model folder contains LSTM and CNN model implemented using pytorch
- Dataloader, NNRegressor is used for training deep learning model

```shell
# Run LSTM-FUSI model
python3 main.py
```

- Report.ipynb include all results and analyzing process
- result.pdf shows the result for all models

### Model Summary

I tried 5 models
- Catboost + TFIDF + SVD
- Catboost + Word2vec/Fasttet
- Catboost + Pretrained Bert
- Catboost + \[TFIDF + SVD, Word2vec, Bert\]
- End-to-end Model - LSTM encoder + time/author/other features embeddings + Fully Connected NN

Result shows LSTM + feature fusing has the best result
The code is written to reproduce the results of DistMult,ComplEx and RESCAL.  
I refer to the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).
Their code is clear and concise.  
I refer the hyperparameters from [OLD DOG](https://openreview.net/forum?id=BkxSmlBFvr).  
For each model,we have hyperparameters which include batch_size,negative_sample_size(usually have the same size as batch_size),embedding_size,gamma,  
learning_rate.The optimizers incluede Adam and Adagrad,which have been integrated into the code.
Some hyperparameters that aren't mentioned in [OLD DOG](https://openreview.net/forum?id=BkxSmlBFvr),I refer them from [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).
  
WN18RR  
bash run.sh train DistMult wn18rr 0 0 1024 1024 512 6.0 0.0004 80000 8

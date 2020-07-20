The code is written to reproduce the results of DistMult, ComplEx and RESCAL.  
I refer to the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).
Their code is clear and concise.  
I refer the hyperparameters from [OLD DOG](https://openreview.net/forum?id=BkxSmlBFvr).
For each model, we have hyperparameters which include batch_size, negative_sample_size, embedding_size, gamma, 
learning_rate and optimizers. The optimizers incluede Adam and Adagrad, which have been integrated into the code. 
Some hyperparameters that aren't mentioned in [OLD DOG](https://openreview.net/forum?id=BkxSmlBFvr), I refer them from [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).  
Thank them for their contributions.
  
WN18RR  
bash run.sh train DistMult wn18rr 0 0 512 1024 512 200.0 0.0005 30000 8  
bash run.sh train ComplEx wn18rr 0 0 512 1024 512 200.0 0.0005 30000 8

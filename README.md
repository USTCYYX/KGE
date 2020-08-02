The code is written to reproduce the results of DistMult, ComplEx and RESCAL.  
I refer to the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).
Their code is clear and concise.  
To improve the performance of the model, I refer the ideas in TransD and TransR, Which think each relation should have a particular semantic space, and I propose the RESCALR, RESCALD, DistMultR and DistMultD. You can see the codes in model.py.  
I refer the hyperparameters from [OLD DOG](https://openreview.net/forum?id=BkxSmlBFvr).
For each model, we have hyperparameters which include batch_size, negative_sample_size, embedding_size, gamma, 
learning_rate and optimizers. The optimizers incluede Adam and Adagrad, which have been integrated into the code. 
Some hyperparameters that aren't mentioned in [OLD DOG](https://openreview.net/forum?id=BkxSmlBFvr), I refer them from [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).  
Thank them for their contributions.
  
WN18RR  
bash run.sh train DistMult wn18rr 0 0 512 1024 512 200.0 0.0005 30000 8 0  
bash run.sh train ComplEx wn18rr 0 0 512 1024 512 200.0 0.0005 30000 8 0  
bash run.sh train RESCAL wn18rr 0 0 512 1024 1024 200.0 0.0005 60000 16 0  
bash run.sh train DistMultR wn18rr 0 0 512 1024 512 200.0 0.0005 30000 8 0  
bash run.sh train DistMultD wn18rr 0 0 512 1024 512 200.0 0.0005 30000 8 0  
bash run.sh train SCE wn18rr 0 0 512 1024 512 200.0 0.0005 30000 8 0  

FB15K-237  
bash run.sh train DistMult FB15K237 0 0 1024 256 512 200.0 0.001 60000 16 0  
bash run.sh train ComplEx FB15K237 0 0 1024 256 512 200.0 0.001 40000 16 0  
bash run.sh train RESCAL FB15K237 0 0 512 256 1024 200.0 0.0005 60000 16 0  
 
bash run.sh train DistMult wn18rr 0 0 512 1024 512 200.0 0.0005 30000 8 0.3  
bash run.sh train ComplEx wn18rr 0 0 512 1024 512 200.0 0.0005 30000 8 0.3  
bash run.sh train RESCAL wn18rr 0 0 512 1024 1024 200.0 0.0005 10000 16 0.4  

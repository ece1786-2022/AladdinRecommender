# AladdinRecommender

As the movie industry has bloomed and developed during the past century, an increasing amount of movies are delivered and published at a fast pace. The role of movie retrieval or recommendation tools is becoming not only more important but also necessary. Even though many movie information database websites(e.g. IMDB) and video streaming service platforms(e.g. Netflix) allow users to search for the desired films/TV series, they are mostly functioned based on predefined input format such as title, genre or regions that users have no flexibility to input arbitrary plots as searching queries. The goal of our project is to construct a content-based recommendation system that accepts informal queries like: “time travel and observe the history” and suggests top movies with the most similar plots. To understand the user inputs and its semantic meaning, natural language processing techniques such as transformers are applicable to this project.

![alt text](https://github.com/ece1786-2022/AladdinRecommender/blob/main/images/Illustration.PNG)


# Data Source and Crawling Scripts
Data Source: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
Crawling Scirpts: crawl/*

# Architecture and Software

The design of our architecture will accept users arbitrary sentence inputs and return the top most relevant movies based on movies complete sentences using minimization of cosine-similarity. The complete sentences and reviews pairs are training data converted to embedding spaces using SBERT (sentence-BERT) model (all-mpnet-base-v2) which contains the configuration of BERT-base, with 12-layer transformer, 12 attention heads, hidden size of 768, token embedding dimension 512, and a total of 110 million parameters. The model is pre-trained MPNet on a 160 GB corpus and maps sentences to a 768 dimensional vector space. Besides, a pooling layer is applied to reduce the dimensions of the hidden layer by averaging the outputs of neuron clusters at the previous layer.
 
The final loss function is OnlineContrastiveLoss which will minimize the distance between the embedding space of complete sentences to positive reviews and maximize the distance between embedding space of complete sentences to negative reviews. To separate the sentiment groups, a sentiment classifier, t5-base-finetuned-imdb-sentiment, containing 220 millions parameters, 32128 vocabulary size, 512 token embedding dimensions, 12 layers transformer and 12 attention heads, is imported to split reviews into positive and negative groups. 

In this way, our model output will be closer to positive reviews and far away from negative results. As the training phase shown below, the OnlineContrastiveLoss is used to update our model through backpropagation and the epochs and batch size are 10 and 4 respectively with default learning rate 2e-05 due to limitation of computation power. 


During the recommendation phase, the users can input anything as sentences which are converted into sentence embedding(y). Then the embedding(y) is compared to all other movies' complete sentences using minimization of cosine-similarity which yields the top most similar movies as our outputs.


Besides, a UI interface is constructed using Gradio embedding our model to the webpage and allowing users to input queries and observe results.

**Code and experiments: transformer/* **

# AladdinRecommender

As the movie industry has bloomed and developed during the past century, an increasing amount of movies are delivered and published at a fast pace. The role of movie retrieval or recommendation tools is becoming not only more important but also necessary. Even though many movie information database websites(e.g. IMDB) and video streaming service platforms(e.g. Netflix) allow users to search for the desired films/TV series, they are mostly functioned based on predefined input format such as title, genre or regions that users have no flexibility to input arbitrary plots as searching queries. The goal of our project is to construct a content-based recommendation system that accepts informal queries like: “time travel and observe the history” and suggests top movies with the most similar plots. To understand the user inputs and its semantic meaning, natural language processing techniques such as transformers are applicable to this project.

![alt text](https://github.com/ece1786-2022/AladdinRecommender/blob/main/images/Illustration.PNG)


# Data Source and Crawling Scripts
Data Source: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

Crawling Scirpts: crawl/\*

# Architecture and Software

The design of our architecture will accept users arbitrary sentence inputs and return the top most relevant movies based on movies complete sentences using minimization of cosine-similarity. The complete sentences and reviews pairs are training data converted to embedding spaces using SBERT (sentence-BERT) model (all-mpnet-base-v2) which contains the configuration of BERT-base, with 12-layer transformer, 12 attention heads, hidden size of 768, token embedding dimension 512, and a total of 110 million parameters. The model is pre-trained MPNet on a 160 GB corpus and maps sentences to a 768 dimensional vector space. Besides, a pooling layer is applied to reduce the dimensions of the hidden layer by averaging the outputs of neuron clusters at the previous layer.

![alt text](https://github.com/ece1786-2022/AladdinRecommender/blob/main/images/Architecture.jpg)

The final loss function is OnlineContrastiveLoss which will minimize the distance between the embedding space of complete sentences to positive reviews and maximize the distance between embedding space of complete sentences to negative reviews. To separate the sentiment groups, a sentiment classifier, t5-base-finetuned-imdb-sentiment, containing 220 millions parameters, 32128 vocabulary size, 512 token embedding dimensions, 12 layers transformer and 12 attention heads, is imported to split reviews into positive and negative groups. 

In this way, our model output will be closer to positive reviews and far away from negative results. As the training phase shown below, the OnlineContrastiveLoss is used to update our model through backpropagation and the epochs and batch size are 10 and 4 respectively with default learning rate 2e-05 due to limitation of computation power. 

![alt text](https://github.com/ece1786-2022/AladdinRecommender/blob/main/images/training_structure.png)

During the recommendation phase, the users can input anything as sentences which are converted into sentence embedding(y). Then the embedding(y) is compared to all other movies' complete sentences using minimization of cosine-similarity which yields the top most similar movies as our outputs.

![alt text](https://github.com/ece1786-2022/AladdinRecommender/blob/main/images/validation_structure.png)

Besides, a UI interface is constructed using Gradio embedding our model to the webpage and allowing users to input queries and observe results.

![alt text](https://github.com/ece1786-2022/AladdinRecommender/blob/main/images/ui.png)

*Code and experiments: transformer/\**

# Baseline Model

After data cleaning and preprocessing, the baseline model supports the same functions as transformers, which accepts users arbitrary sentence inputs and returns the top most relevant movies based on input queries. Our baseline model mainly uses tf-idf (term-frequency and inverse-document-frequency) scores to vectorize all movie information such as plot, genres, etc and form the tf-idf matrix. The recommender will iterate through the matrix of vectors and find the top most similar movies vectors  given user input queries using minimization of cos-similarity. Since tf-idf vectorizers have less concern about the sentences ordering but concentrate more on tokens, all relative information could be added at tails. Besides, n-grams are used for lengths of 1 to 3 in order to expand our corpus size and extract more combinations of tokens. 

*Code and experiments: baselines/\**

# Discussion and Learnings

The model performance is surprisingly well based on the results from both quantitative and qualitative results, especially the high values of MAP. One thing we noticed during the development of our model is that because of how informal queries have vague meanings compared to solid-feature queries, people could interpret them differently. This means no matter how accurately our model performs, personalized preference setting should be considered for users. Even if this is a common situation for a pure content-based recommendation system without considering user preferences, it could be certainly improved by combining the session-based or collaborative-based methods if we start again by shifting some concentration from NLP applications to results improvements and availability out of this course. Besides, a more comprehensive dataset would be much useful that more movies can be trained on instead of filtered out without enough information The best usage scenario for our model could be embedded into the search engine of a movie website, we are allowed to not only apply some hard constraint to the searching result(e.g. Language, R-rating), but also adjust the model based on the user's choice of recommendations. 

# Reference
1. J. Ng, “Content-based recommender using Natural Language Processing (NLP),” KDnuggets. [Online]. Available: https://www.kdnuggets.com/2019/11/content-based-recommender-using-natural-language-processing-nlp.html.
2. G. Moreira, “Transformers4Rec: A flexible library for sequential and session-based recommendation,” Medium, 08-Nov-2021. [Online]. Available: https://medium.com/nvidia-merlin/transformers4rec-4523cc7d8fa8
3. T. M. D. (TMDb), “TMDB 5000 movie dataset,” Kaggle, 28-Sep-2017. [Online]. Available: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
4. N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using Siamese Bert-Networks,” arXiv.org, 27-Aug-2019. [Online]. Available: https://arxiv.org/abs/1908.10084
5. O. Espejel, “Sentence-transformers/all-mpnet-base-V2 · hugging face,” sentence-transformers/all-mpnet-base-v2 · Hugging Face. [Online]. 6. Available: https://huggingface.co/sentence-transformers/all-mpnet-base-v2.
6. M. Romero, “MRM8488/T5-base-finetuned-imdb-sentiment · hugging face,” mrm8488/t5-base-finetuned-imdb-sentiment · Hugging Face. [Online]. Available: https://huggingface.co/mrm8488/t5-base-finetuned-imdb-sentiment


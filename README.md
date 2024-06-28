# FactChecker

## Authors: Vinayak (Vin) Kannan

## Description:
This repository contains the code for an explainable fact-checking system.
Our method is detailed below:

We seek to determine if an explainable approach leveraging deep learning techniques can be applied to the fact-checking domain. 
We define an explainable method as one that gives the end user insight as to why the model classified a claim as true / false.
This explainability is critical in order to earn users’ trust in the system.

To this end, we’ve developed a four-stage pipeline to create an explainable fact-checking system.
The pipeline is broken down into 4 stages: ClaimBuster Filtration, LLM Embedding Generation, Supervised Dimensionality Reduction, and Recursive HDBScan.
Inspired by topic modeling pipelines, our system’s output is a claim’s predicted veracity label and an assigned topic cluster.
This provides the user with an explanation behind the predicted veracity label
(e.g., “We believe this claim is true / false because other similar claims like it tend to be true / false”). To focus our model, we have restricted our method to presently tackle claims related to climate change in English made between 2019 to 2024.

Details for each stage of the pipeline are as follows:

1. ClaimBuster Filtration 
- When using scraped online claims to train our pipeline, our team found it critical to devise a method to distinguish which claims are worth fact-checking. As seen in Figure Z, some claims may simply not be ‘controversial’ enough to warrant further examination. In turn, we needed to create a filtration mechanism to ensure our pipeline only examines the correct claims from online sources. 
- So, the first stage of our pipeline accomplishes this filtration task through two steps: sentence tokenization and ClaimBuster check-worthiness scoring.
- The first step, sentence tokenization, is performed using NLTK’s PunktSentenceTokenizer. This tokenization is done to properly leverage ClaimBuster’s ClaimSpotter technique in the second step of our technique.
- For context, ClaimSpotter is a semantic classifier that uses tokens and POS tagging to classify a sentence as ‘check-worthy’. This classifier outputs a score from 0.0 to 1.0; ‘The higher the score, the more likely the sentence contains check-worthy factual claims’ [https://ranger.uta.edu/~cli/pubs/2017/claimbuster-vldb17demo-hassan.pdf]. Using this score, we then apply a heuristic filter to remove any claims that are not worth fact-checking.

2. LLM Embedding Generation 
- The second stage of the pipeline is generating vector embedding representations of each check-worthy claim. This step is done for our labeled training dataset, where each claim is labeled as true / false, and our test set of claims. In this stage, we used OpenAI’s text-embedding-3-large embeddings to generate vector representations and store these embeddings in Chroma DB for further retrieval for topic modeling. We also explore using different models to generate embeddings and the performance tradeoffs in the Experiments section of this paper.

3. Supervised Dimensionality Reduction 
- The third stage of the pipeline leverages supervised UMAP to reduce the dimensionality of the vector embeddings to a lower dimensionality appropriate for clustering algorithms. Using UMAP, we can reconstruct the vector representations of our check-worthy claims in a lower dimension. Simultaneously, we can also leverage supervised UMAP as a form of Metric Learning to assist with classifying true / false claims. This Metric Learning stage is pivotal, as it enables our system to both learn from user-provided labels for future claims and adapt to changes in veracity labels for existing claims in our pipeline’s training dataset [https://arxiv.org/abs/1802.03426]. To assist with the last stage of the pipeline, we store the reduced dimensionality vectors in Chroma DB for future retrieval. We explore hyperparameter selection for UMAP (e.g., number of neighbors, supervised versus unsupervised, etc.) and the impact of skipping this stage in the Experiments section of the paper.


4. Recursive HDBScan 
- Finally, the fourth stage of our pipeline uses HDBScan to cluster our reduced dimensionality vectors into topics. We found that leveraging a recursive reclustering algorithm that looks to create ‘pure veracity clusters’, as depicted in Figure A, leads to the best performance; we detail these experiment trails in the Experiments section of this paper. Pure veracity clusters, for context, are topic clusters where the majority label of training dataset claims, within each cluster, constitutes over a certain threshold percentage. As detailed in Figure A, we recursively run HDBScan within each cluster until this threshold percentage is reached across all clusters in the dataset. 
- Using these clusters, we then perform weighted KNN for each claim that we want to predict the veracity for; this algorithm is detailed in Figure B. Note that the hyperparameter K in our weighted KNN is set to include all the claims within the test claim’s individual cluster. Afterward, we return both the predicted veracity label and the associated cluster for the test claim to the user. This provides an explainable logic to the user, detailing that our system believes the user’s claim is true / false due to its similarity to other claims with the same veracity label

## How to use this repository:
1. Ask Vin to join the Pinecone Organization. Get API keys for both Pinecone, OpenAI, ClaimBuster, and Reddit to fill out .env file in root of project
2. Create a venv and install the requirements.txt file. Use python version 3.11
2. Navigate to hyperparameter_selection and run all the cells to begin an experiment
3. To adjust the parameters in the experiment, go to ParameterCreator.py and add values to the arrays. This class creates parameters to assist with grid search
4. The hyperparameter_selection code will output the results or each experiment in a table in the last cell. Download this table to view the results


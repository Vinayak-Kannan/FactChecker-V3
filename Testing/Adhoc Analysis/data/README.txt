-----------------------------------------------------
Data used in Coan, Boussalis, Cook, and Nanko (2021)
-----------------------------------------------------

This directory includes two sub-directories that house the main
data used during training and in the analysis.

------------------
analysis directory
------------------

The analysis directory includes a single CSV file: cards_for_analysis.csv. The
file has the following fields:

domain: the domain for each organization or blog.

date: the date the article or blog post was written.

ctt_status: an indictor for whether the source is a conservative think tank
(CTTs). [CTT = True; Blog = False]

pid: unique paragraph identifier

claim: the estimated sub-claim based on the RoBERTa-Logistic ensemble described
in the paper. [The variable assumes the following format: superclaim_subclaim.
For example, 5_1 would represent super-claim 5 ("Climate movement/science is
unreliable"), sub-claim 1 ("Science is unreliable").]

------------------
training directory
------------------

The training directory includes 3 CSV files:

training.csv: annotations used for training
validation.csv: the held-out validation set used during training (noisy)
test.csv: the held-out test set used to assess final model performance
(noise free)

Each file has the following fields:

text: the paragraph text that is annotated
claim: the annotated sub-claim [The variable assumes the following format:
superclaim_subclaim. For example, 5_1 would represent super-claim 5
("Climate movement/science is unreliable"), sub-claim 1 ("Science is
unreliable").]

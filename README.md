# skip-gram-w2v
Skip gram implementation of word-2-vector for NLU Assignment 1.

In this implementation, reuters data is stored in the pickle file.

Data files:
reuters_train.pkl\n
reuters_test.pkl\n
reuters_processed_train.pkl\n

Evaluation files:
EN-SIMLEX-999.txt = simplex99 similiarity file
question-words.txt = data for Analogy Task

Code files:
garbage.txt = contains some garbage text which is removed during preprocessing step
preprocess.py = preprocessing code
w2v_np = model training code
wordsim.py, ranking.py, read_write = code to calculate correlation coefficient, directly picked up from https://github.com/mfaruqui/eval-word-vectors

Train:
   python w2v_np.py reuters_processed_train.pkl

Embedding file is created after the process. Each line in the embedding file contain a word and its vector components, all space seperated.
eg  the 0.756 2.484 0.0345 ...

to use embedding for downstream task, use read_write.py

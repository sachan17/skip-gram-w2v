# skip-gram-w2v
Skip gram implementation of word-2-vector for NLU Assignment 1.

In this implementation, reuters data is stored in the pickle file.

Data files:<br/>
reuters_train.pkl<br/>
reuters_test.pkl<br/>
reuters_processed_train.pkl<br/>

Evaluation files:<br/>
EN-SIMLEX-999.txt = simplex999 similiarity file<br/>
question-words.txt = data for Analogy Task<br/>

Code files:<br/>
garbage.txt = contains some garbage text which is removed during preprocessing step<br/>
preprocess.py = preprocessing code<br/>
w2v_np = model training code<br/>
wordsim.py, ranking.py, read_write = code to calculate correlation coefficient, directly picked up from https://github.com/mfaruqui/eval-word-vectors<br/>

Train:<br/>
&nbsp;&nbsp;&nbsp; `python w2v_np.py reuters_processed_train.pkl`

Embedding file is created after the process. Each line in the embedding file contain a word and its vector components, all space seperated.<br/>
eg: &nbsp;the 0.756 2.484 0.0345 ...

Test:<br/>
for simplex999:<br/>
&nbsp;&nbsp;&nbsp;`python wordsim.py Embeddings_3_0.03.txt EN-SIMLEX-999.txt`<br/>
for Analogy task: <br/>
&nbsp;&nbsp;&nbsp;`python analogy_eval.py Embeddings_3_0.03.txt`

To use embedding for downstream task, use read_write.py

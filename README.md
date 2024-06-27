![CARMA 2024 Conference](https://github.com/istat-methodology/TopicModelingLab/blob/main/resources/carma2024.png)

<h1 align="center">
  Topic Modeling Tutorial
</h1>
<div align="center">
  
  <a href="">![Static Badge](https://img.shields.io/badge/LDA-red)</a>
  <a href="">![Static Badge](https://img.shields.io/badge/HDP-blue)</a>
  <a href="">![Static Badge](https://img.shields.io/badge/NMF-green)</a>
  <a href="">![Static Badge](https://img.shields.io/badge/Top2Vec-cian)</a>
  <a href="">![Static Badge](https://img.shields.io/badge/BERTopic-yellow)</a>
  
</div>

<p align="center">
  A review of the most popular topic modeling techniques.
</p>

<div align="center">
  
  <a href="https://www.researchgate.net/profile/Mauro-Bruno-2">
  <img src="https://img.shields.io/badge/Mauro%20Bruno-white?logo=researchgate" alt="Static Badge">
</a>
  <a href="">![Static Badge](https://img.shields.io/badge/Elena%20Catanese-white?logo=researchgate&link=https%3A%2F%2Fwww.researchgate.net%2Fprofile%2FElena-Catanese-2)</a>
  <a href="">![Static Badge](https://img.shields.io/badge/Francesco%20Ortame-white?logo=researchgate&link=https%3A%2F%2Fwww.researchgate.net%2Fprofile%2FFrancesco-Ortame-3)</a>
  <a href="">![Static Badge](https://img.shields.io/badge/Francesco%20Pugliese-white?logo=researchgate&link=https%3A%2F%2Fwww.researchgate.net%2Fprofile%2FFrancesco-Pugliese-9)</a>

</div>

<div align="center">
  
  <a href="">![GitHub Repo stars](https://img.shields.io/github/stars/istat-methodology/TopicModelingLab?style=for-the-badge&logo=github)</a>

</div>

---

## What is Topic Modeling?
Topic modeling is a type of statistical modeling used to discover abstract topics within a collection of documents. This technique is widely used in natural language processing (NLP) to uncover hidden patterns and structure in large textual datasets. The primary goal of topic modeling is to automatically identify topics present in a corpus and to organize the documents according to these topics.

### A few concepts...
* 📄 **Documents and Words**: The basic units of topic modeling. Documents are the individual pieces of text (e.g., Tweets, reviews...), and words are the tokens or terms within these documents.
* 📚 **Corpus**: A corpus is a collection of documents. Topic modeling algorithms analyze the corpus to identify the topics.
* 🗯️ **Topics**: A topic is a distribution over a fixed vocabulary. It is characterized by a set of words that frequently appear together. Each topic can be seen as a pattern of co-occurrence of words.
* 👻 **Latent Variables**: These are variables that are not directly observed but are inferred from other variables that are observed (e.g., the words in documents).

### Steps in topic modeling

![Topic Modeling Pipeline](https://github.com/istat-methodology/TopicModelingLab/blob/main/resources/topic-modeling-pipeline.png)

1. 🧼 **Pre-processing**: Clean the text data (e.g., remove stopwords and tokenize)
2. 🏋️ **Model Training**: Apply a topic modeling algorithm to the preprocessed data
3. 📈 **Evaluation**: Assess the quality of the topics using coherence scores, human judgement, or other metrics
4. ✍️ **Interpretation**: Analyze the topics and assign meaningful labels or descriptions

---

## Traditional Topic Modeling Techniques
<div>

  <a href="">![Static Badge](https://img.shields.io/badge/LDA-red)</a>
  <a href="">![Static Badge](https://img.shields.io/badge/HDP-blue)</a>
  <a href="">![Static Badge](https://img.shields.io/badge/NMF-green)</a>
  
</div>
Traditional topic modeling techniques rely on statistical methods to uncover hidden topics in a corpus. We will explore Latent Dirichlet Allocation (LDA), Hierarchical Dirichlet Process (HDP), and Non-negative Matrix Factorization (NMF).  

```python
from gensim.models import LdaModel, HdpModel, Nmf
```

### 🧼 Pre-processing
Before training our models, we need to ensure that the data is in the correct format.

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora

# download the stopwords for the reference language
nltk.download('stopwords')
nltk.download('punkt')

language = 'italian'
stop_words = set(stopwords.words(language))

# clean the texts
clean_sentences = []

for document in sentences:
    tokens = word_tokenize(document)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    clean_sentences.append(filtered_tokens)

# dictionary and bag-of-words (BOW)
dictionary = corpora.Dictionary(clean_sentences)
corpus_bow = [dictionary.doc2bow(t) for t in clean_sentences]
```

---

### Latent Dirichlet Allocation (LDA)
Latent Dirichlet Allocation (LDA) is a generative probabilistic model that assumes documents to be mixtures of topics, and topics to be mixtures of words.

LDA involves the following steps:
1. **Parameter Initialization**: LDA initializes the topics, the topic distribution for each document, and the word distribution for each topic.
2. **Training**: Using an iterative process (usually Gibbs sampling or variational inference), LDA refines these distributions to better fit the observed data.
3. **Topic Inference**: After several iterations, LDA infers the topic distribution for each document and the word distribution for each topic.

In LDA, the number of latent topics to be extracted needs to be defined *a priori*.

To implement LDA in python using the `gensim` library, we simply run:

```python
# train the LDA model
lda_model = LdaModel(
    corpus=corpus_bow,
    id2word=dictionary,
    passes = 10,
    num_topics=20
)

# visualize the extracted topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}\n")
```
---

### **Hierarchical Dirichlet Process (HDP)**
Hierarchical Dirichlet Process (HDP) is a non-parametric Bayesian approach to topic modeling. Unlike LDA, HDP automatically determines the number of topics based on the data.

HDP involves the following steps:
1. **Initialization**: HDP starts with an initial guess on the topic distribution.
2. **Iterative Refinement**: Using a hierarchical process, HDP refines the topic distribution at both the document and corpus levels.
3. **Dynamic Topic Adjustment**: HDP adjusts the number of topics dynamically as more data is processed.

To implement HDP in python using the `gensim` library, we simply run:

```python
# train the HDP model
hdp_model = HdpModel(
    corpus=corpus_bow,
    id2word=dictionary
)

# visualize the extracted topics
for idx, topic in hdp_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}\n")
```

---

### **Non-negative Matrix Factorization (NMF)**
Non-negative Matrix Factorization (NMF) is a non-probabilistic technique, particularly suitable for large, sparse datasets. Unlike probabilistic models like LDA and HDP, NMF is a linear algebra-based method that decomposes the document-term matrix into two lower-dimensional matrices, one representing the topics and the other representing the topic distribution for each document.

NMF involves the following steps:
1. **Matrix Decomposition**: NMF decomposes the document-term matrix into two non-negative matrices.
2. **Iterative Optimization**: Using iterative optimization techniques, NMF refines these matrices to minimize the reconstruction error.
3. **Topic Extraction**: The resulting matrices are used to extract the topics and their distribution across documents.

Likewise LDA, NMF requires the number of topics to extract to be defined *a priori*.

To implement NMF in python using the `gensim` library, we simply run:

```python
# train the NMF model
nmf_model = Nmf(
    corpus=corpus_bow,
    id2word=dictionary,
    num_topics=20
)

# visualize the extracted topics
for idx, topic in nmf_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}\n")
```

---

### Comparison of LDA, HDP, and NMF
| Criteria | LDA | HDP | NMF |
| --- | --- | --- | --- |
| **Model Type** | Probabilistic | Probabilistic | Linear Algebra |
| **# Topics** | Pre-specified | Dynamic | Pre-specified |
| **Scalability** | Good | Moderate | Good |

---

## Clustering Algorithms on Embedding Spaces
<div>

  <a href="">![Static Badge](https://img.shields.io/badge/Top2Vec-cian)</a>
  <a href="">![Static Badge](https://img.shields.io/badge/BERTopic-yellow)</a>
  
</div>
Unlike traditional techniques, clustering algorithms on embedding spaces leverage neural network-based word embeddings to discover latent topics in text data. These approaches rely on dense word representations to capture semantic relationships more effectively. We will explore the Top2Vec and BERTopic algorithms.

```python
from top2vec import Top2Vec
from bertopic import BERTopic
```

### **Top2Vec**
Top2Vec is an algorithm that simultaneously learns the topic representations and the word embeddings. By mapping documents to a continuous vector space, Top2Vec identifies clusters of documents that share similar themes without requiring a pre-defined number of topics. This approach allows for the discovery of natural and meaningful topics directly from the data.

Top2Vec involves the following steps:
1. **Embedding Creation**: Top2Vec uses word embeddings to create document vectors.
2. **Dimensionality Reduction**: The document vectors are reduced to a lower-dimensional space using techniques like UMAP.
3. **Clustering**: The reduced vectors are clustered to identify topics.
4. **Topic Words Identification**: The algorithm finds words that are closest to the cluster centroids, representing the topics.

To implement Top2Vec in python using the `top2vec` library, we can choose a pre-trained embedding model and run:

```python
# choose the embedding model
embeddings = "distiluse-base-multilingual-cased"

# train the Top2Vec model
top2vec_model = Top2Vec(
    sentences,
    embedding_model = embeddings
)

# visualize the extracted topics
topics = top2vec_model.get_topics()

for topic in topics:
    topic_words, word_scores, topic_scores, topic_num = topic
    print(f"Topic {topic_num}:")
    print(f"Words: {topic_words} \nScores: {word_scores}\n")
```

---

### **BERTopic**
BERTopic leverages transformer-based embeddings to create document representations and applies clustering algorithms to discover topics. It combines BERT embeddings with clustering techniques like HDBSCAN and dimensionality reduction methods like UMAP to generate coherent topics from text data.

BERTopic involves the following steps:
1. **Embedding Creation**: BERTopic uses transformer models to create document embeddings.
2. **Dimensionality Reduction**: The embeddings are reduced in dimensionality using UMAP.
3. **Clustering**: Density-based clustering algorithm such as HDBSCAN are applied to form clusters from the reduced embeddings.
4. **Topic Representation**: The algorithm generates topics based on the clustered documents and their embeddings. In this step, LLMs can be used for automatic and meaningful topic labeling.

To implement BERTopic in python using the `bertopic` library, we can choose a pre-trained embedding model and run:

```python
from sentence_transformers import SentenceTransformer

# import the embedding model
embeddings = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# train the BERTopic model
BERTopic_model = BERTopic(
    embedding_model=embeddings,
    min_topic_size=30
)
topics, probs = BERTopic_model.fit_transform(sentences)

# visualize the extracted topics
print(BERTopic_model.get_topic_info())
```

import tensorflow as tf
import tensorflow_text  # Required for Universal Sentence Encoder
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

# Load the text
text = "Your input text goes here."

# Sentence tokenization
sentence = tf.strings.unicode_split(tf.constant(text), 'UTF-8').numpy().tolist()

# Cleaning the sentences
corpus = []
for i in range(len(sentence)):
    sen = re.sub('[^a-zA-Z]', " ", sentence[i].decode('utf-8'))
    sen = sen.lower()
    sen = sen.split()
    sen = ' '.join([i for i in sen if i not in tf.keras.preprocessing.text.text_to_word_sequence(' '.join(stopwords.words('english')))])
    corpus.append(sen)

# Load Universal Sentence Encoder
embed = tf.keras.layers.Embedding(1, 512, input_length=len(corpus))(tf.constant(corpus))
use_model = tf.keras.Model(inputs=[embed], outputs=[embed])

# Generate sentence vectors using Universal Sentence Encoder
sen_vectors = use_model.predict(embed)

# Create cosine similarity matrix
sim_mat = cosine_similarity(sen_vectors, sen_vectors)

# Normalize the similarity matrix
norm = np.sum(sim_mat, axis=1)
sim_mat = np.divide(sim_mat, norm[:, np.newaxis], where=norm[:, np.newaxis] != 0)

# Create a network graph using cosine similarity
G = nx.from_numpy_array(np.array(sim_mat))
nx.draw(G, with_labels=True)
plt.title('Cosine Similarity Graph')
plt.show()

# Calculate TextRank scores for each sentence
scores = nx.pagerank_numpy(G)

# Print the top k sentences based on TextRank scores
k = int(input('Enter the number of sentences to extract: '))
ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentence)), reverse=True)

selected_sentences = [s for _, s in ranked_sentences[:k]]

# Print the selected sentences
print(f"\nTop {k} sentences based on TextRank scores:")
for s in selected_sentences:
    print(s.decode('utf-8'))

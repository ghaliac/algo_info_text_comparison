import nltk
from nltk.corpus import brown
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import ssl
import numpy as np
import string
import gzip
from scipy.optimize import curve_fit
import json
import os
import random
from math import log2

# Gestion des certificats SSL pour le téléchargement
try:
    _create_unverified_https_context = ssl._create_unverified_context
    print("Configuration SSL ajustée pour un contexte non vérifié.")
except AttributeError:
    print("Échec de la modification de la configuration SSL, la fonction n'existe pas.")
else:
    ssl._create_default_https_context = _create_unverified_https_context
    print("Contexte SSL par défaut établi pour une vérification non stricte.")

# Télécharger les ressources nécessaires
try:
    nltk.download('brown')
    nltk.download('punkt', force=True)
    nltk.download('stopwords')
    print("Ressources NLTK téléchargées avec succès.")
except Exception as e:
    print(f"Erreur lors du téléchargement des ressources NLTK : {e}")

# Sélectionner les mots du corpus Brown
words = brown.words()
print(f"Nombre de mots dans le corpus Brown: {len(words)}")

# Calculer les fréquences des mots
fdist = FreqDist(words)
print(f"Nombre de mots uniques dans le corpus Brown: {len(fdist)}")

# Loi de Heaps
def heaps_law(x, k, beta):
    return k * x ** beta

# Fonctions utilitaires
def extract_vocabulary_size(corpus):
    words = nltk.word_tokenize(corpus)
    fdist = FreqDist(words)
    return len(fdist)

def measure_compression_ratio(corpus_text):
    text_bytes = corpus_text.encode('utf-8')
    compressed = gzip.compress(text_bytes)
    return len(compressed) / len(text_bytes)

def calculate_entropy(corpus_text):
    words = nltk.word_tokenize(corpus_text)
    fdist = FreqDist(words)
    total_words = len(words)
    probabilities = [freq / total_words for freq in fdist.values()]
    entropy = -sum(p * log2(p) for p in probabilities)
    return entropy

def add_top_words_box(ax, corpus_name, corpus_text, top_n=10):
    words = nltk.word_tokenize(corpus_text)
    fdist = FreqDist(words)
    most_common_words = fdist.most_common(top_n)
    text = f"Top {top_n} mots pour {corpus_name}:\n" + "\n".join([f"{rank+1}. {word} ({freq})" for rank, (word, freq) in enumerate(most_common_words)])
    ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

def analyze_corpus(corpus_name, corpus_text):
    print(f"Commence l'analyse du corpus: {corpus_name}")
    corpus_sizes = []
    vocab_sizes = []
    step_size = 100000
    for i in range(0, len(corpus_text), step_size):
        subset = corpus_text[:i + step_size]
        corpus_sizes.append(len(subset))
        vocab_sizes.append(extract_vocabulary_size(subset))
        print(f"Taille du corpus analysé jusqu'à {i + step_size} caractères.")
    
    log_corpus_sizes = np.log(corpus_sizes)
    log_vocab_sizes = np.log(vocab_sizes)

    params, _ = curve_fit(lambda x, a, b: a + b * x, log_corpus_sizes, log_vocab_sizes)
    log_k, beta = params
    k = np.exp(log_k)
    
    print(f"Paramètres estimés (log-log) pour {corpus_name}: K = {k:.2f}, β = {beta:.2f}")
    
    compression_ratio = measure_compression_ratio(corpus_text)
    print(f"Ratio de compression pour {corpus_name} : {compression_ratio:.2f}")

    entropy = calculate_entropy(corpus_text)
    print(f"Entropie pour {corpus_name} : {entropy:.2f}")
    return corpus_sizes, vocab_sizes, k, beta

# Génération de textes synthétiques et mesure de la croissance du vocabulaire
print("Génération de textes synthétiques...")
text_lengths = np.linspace(1000, len(words), 50)
vocab_sizes = []
corpus_sizes = []

for index, length in enumerate(text_lengths):
    synthetic_text = ' '.join(random.choices(population=list(fdist.keys()), weights=list(fdist.values()), k=int(length)))
    synthetic_words = nltk.word_tokenize(synthetic_text)
    synthetic_fdist = FreqDist(synthetic_words)
    vocab_sizes.append(len(synthetic_fdist))
    corpus_sizes.append(length)
    print(f"Étape {index+1}/{len(text_lengths)}: Taille du corpus généré = {length}, Taille du vocabulaire = {len(synthetic_fdist)}")

corpus_name = "Synthetic Text"
corpus_sizes, vocab_sizes, k, beta = analyze_corpus(corpus_name, synthetic_text)

plt.figure(figsize=(10, 6))
plt.scatter(corpus_sizes, vocab_sizes, color='blue', label='Données observées')
plt.plot(corpus_sizes, [heaps_law(x, k, beta) for x in corpus_sizes], 'r--', label=f'Ajustement: k={k:.2f}, beta={beta:.2f}')
plt.xlabel('Taille du Corpus')
plt.ylabel('Taille du Vocabulaire')
plt.title('Loi de Heaps - Corpus Brown Synthétique')
plt.legend()
plt.grid(True)
plt.show()

def zipf_linear_fit(x, a, b):
    return a + b * x

def analyze_zipf(corpus_name, corpus_text):
    tokens = nltk.word_tokenize(corpus_text)
    fdist = FreqDist(tokens)
    frequencies = sorted(fdist.values(), reverse=True)[:100]  # Limiter aux 100 premiers rangs
    
    ranks = np.arange(1, len(frequencies) + 1)
    log_ranks = np.log(ranks)
    log_freqs = np.log(frequencies)
    
    # Utilisation de curve_fit pour ajuster la fonction linéaire
    params, covariance = curve_fit(zipf_linear_fit, log_ranks, log_freqs)
    a, b = params  # a est l'intercept, b est la pente

    # Prédiction à partir des paramètres ajustés
    predicted_log_freqs = zipf_linear_fit(log_ranks, *params)
    
    # Calcul du score R2
    residuals = log_freqs - predicted_log_freqs
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_freqs - np.mean(log_freqs))**2)
    r2_score = 1 - (ss_res / ss_tot)
    
    # Tracer les résultats en échelle linéaire
    plt.figure(figsize=(10, 6))
    plt.scatter(ranks, frequencies, label='Données')
    plt.plot(ranks, np.exp(predicted_log_freqs), 'r', label=f'Ligne ajustée (pour échelle loglog): pente={b:.2f}')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title(f'Analyse de la loi de Zipf {corpus_name}')
    plt.legend()
    plt.grid(True)
    
    # Ajouter les mots des 5 premiers rangs dans une boîte
    most_common_words = fdist.most_common(5)
    text = f"Top 5 mots pour {corpus_name}:\n" + "\n".join([f"{rank+1}. {word} ({freq})" for rank, (word, freq) in enumerate(most_common_words)])
    plt.gca().text(0.95, 0.95, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                   horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.show()
    
    print(f"Intercept: {a}")
    print(f"Slope: {b}")
    print(f"R2 Score: {r2_score:.2f}")

analyze_zipf("Synthetic Text", synthetic_text)

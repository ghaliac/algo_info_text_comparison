import nltk
from nltk.corpus import brown
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import random
import ssl
import numpy as np
from scipy.optimize import curve_fit

# Gestion des certificats SSL pour le téléchargement
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

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

# Générer un texte synthétique et mesurer la croissance du vocabulaire
text_lengths = np.linspace(1000, len(words), 50)
vocab_sizes = []
corpus_sizes = []

print("Génération de textes synthétiques et mesure de la croissance du vocabulaire...")
for index, length in enumerate(text_lengths):
    synthetic_text = ' '.join(random.choices(population=list(fdist.keys()), weights=list(fdist.values()), k=int(length)))
    synthetic_words = nltk.word_tokenize(synthetic_text)
    synthetic_fdist = FreqDist(synthetic_words)
    vocab_sizes.append(len(synthetic_fdist))
    corpus_sizes.append(length)
    print(f"Étape {index+1}/{len(text_lengths)}: Taille du corpus généré = {length}, Taille du vocabulaire = {len(synthetic_fdist)}")

# Tracer la loi de Heaps
print("Ajustement de la loi de Heaps...")
params, _ = curve_fit(heaps_law, corpus_sizes, vocab_sizes)
k, beta = params

plt.figure(figsize=(10, 6))
plt.plot(corpus_sizes, vocab_sizes, 'o', label='Données observées')
plt.plot(corpus_sizes, heaps_law(np.array(corpus_sizes), *params), '-', label=f'Heaps Law Fit: k={k:.2f}, beta={beta:.2f}')
plt.xlabel('Taille du Corpus')
plt.ylabel('Taille du Vocabulaire')
plt.title('Loi de Heaps - Corpus Brown Synthétique')
plt.legend()
plt.grid(True)
plt.show()

# Fonction pour tracer la loi de Zipf
def plot_zipf_law(text):
    print("Tracé de la loi de Zipf...")
    words = nltk.word_tokenize(text)
    fdist = FreqDist(words)
    ranks = range(1, len(fdist) + 1)
    max_rank = 100
    sorted_frequencies = sorted(fdist.values(), reverse=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(ranks[:max_rank], sorted_frequencies[:max_rank], marker='.', linestyle='None')
    plt.title('Loi de Zipf du Texte Synthétique - Corpus Brown')
    plt.xlabel('Rang du mot')
    plt.ylabel('Fréquence')
    plt.grid(True)
    plt.show()

# Tracer la loi de Zipf pour un texte synthétique de taille maximale
plot_zipf_law(' '.join(random.choices(population=list(fdist.keys()), weights=list(fdist.values()), k=len(words))))

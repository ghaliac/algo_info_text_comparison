import nltk
from nltk.corpus import gutenberg, brown, reuters
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import ssl
import numpy as np
from scipy.optimize import curve_fit


# Fonction pour extraire la taille du vocabulaire
def extract_vocabulary_size(corpus):
    words = nltk.word_tokenize(corpus)
    fdist = FreqDist(words)
    return len(fdist)


# Loi de Heaps
def heaps_law(x, k, beta):
    return k * x**beta


# Gestion des certificats SSL pour le téléchargement
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Télécharger les ressources nécessaires
try:
    nltk.download('gutenberg')
    nltk.download('brown')
    nltk.download('reuters')
    nltk.download('punkt', force=True)
except Exception as e:
    print(f"Erreur lors du téléchargement des ressources NLTK : {e}")


# Fonction pour analyser un corpus
def analyze_corpus(corpus_name, corpus_text, step_size=100000):
    print(f"\n--- Analyse du corpus : {corpus_name} ---")
    corpus_sizes = []
    vocab_sizes = []

    for i in range(0, len(corpus_text), step_size):
        subset = corpus_text[:i + step_size]  # Extraction d'un sous-ensemble du texte
        corpus_size = len(subset)
        vocab_size = extract_vocabulary_size(subset)
        corpus_sizes.append(corpus_size)
        vocab_sizes.append(vocab_size)
        print(f"Taille du corpus : {corpus_size}, Taille du vocabulaire : {vocab_size}")

    # Régression log-log
    log_corpus_sizes = np.log(corpus_sizes)
    log_vocab_sizes = np.log(vocab_sizes)

    def linear_fit(x, a, b):
        return a + b * x

    params, _ = curve_fit(linear_fit, log_corpus_sizes, log_vocab_sizes)
    log_k, beta = params
    k = np.exp(log_k)  # Convertir log(K) en K

    print(f"Paramètres estimés (log-log) pour {corpus_name} : K = {k:.2f}, β = {beta:.2f}")

    return corpus_sizes, vocab_sizes, k, beta


# Initialiser une figure pour afficher plusieurs corpus
plt.figure(figsize=(10, 6))

# Analyser et tracer chaque corpus
corpora = {
    "Gutenberg (Littéraire)": ''.join([gutenberg.raw(text) for text in [
        'austen-emma.txt', 'shakespeare-hamlet.txt', 'melville-moby_dick.txt',
        'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt',
        'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt'
    ]]),
    "Brown (Généraliste/Conversationnel)": ' '.join(brown.words()),
    "Reuters (Journalistique)": ' '.join(reuters.words())
}

for corpus_name, corpus_text in corpora.items():
    corpus_sizes, vocab_sizes, k, beta = analyze_corpus(corpus_name, corpus_text)
    
    # Tracer les données réelles
    plt.loglog(corpus_sizes, vocab_sizes, marker='o', linestyle='-', label=f'{corpus_name} (Données réelles)')
    
    # Tracer la courbe ajustée
    fitted_vocab_sizes = [heaps_law(x, k, beta) for x in corpus_sizes]
    plt.loglog(corpus_sizes, fitted_vocab_sizes, linestyle='--', label=f'{corpus_name} (Ajustement: K={k:.2f}, β={beta:.2f})')

# Finaliser le graphique
plt.xlabel('Taille du corpus (log)')
plt.ylabel('Taille du vocabulaire (log)')
plt.title('Comparaison de la loi de Heaps entre différents corpus')
plt.legend()
plt.grid(True)
plt.show()

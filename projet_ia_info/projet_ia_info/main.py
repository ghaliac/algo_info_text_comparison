import nltk
from nltk.corpus import gutenberg, brown, reuters
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import ssl
import numpy as np
from scipy.optimize import curve_fit
import gzip
import json
import os
from math import log2


# Fonction pour extraire la taille du vocabulaire
def extract_vocabulary_size(corpus):
    words = nltk.word_tokenize(corpus)
    fdist = FreqDist(words)
    return len(fdist)


# Loi de Heaps
def heaps_law(x, k, beta):
    return k * x**beta


# Calculer le ratio de compression
def measure_compression_ratio(corpus_text):
    text_bytes = corpus_text.encode('utf-8')  # Convertir en bytes
    compressed = gzip.compress(text_bytes)  # Compression avec gzip
    return len(compressed) / len(text_bytes)  # Ratio de compression


# Calculer l'entropie de Shannon
def calculate_entropy(corpus_text):
    words = nltk.word_tokenize(corpus_text)
    fdist = FreqDist(words)
    total_words = len(words)
    probabilities = [freq / total_words for freq in fdist.values()]
    entropy = -sum(p * log2(p) for p in probabilities)
    return entropy


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

    # Calculer le ratio de compression pour le corpus complet
    compression_ratio = measure_compression_ratio(corpus_text)
    print(f"Ratio de compression pour {corpus_name} : {compression_ratio:.2f}")

    # Calculer l'entropie de Shannon pour le corpus complet
    entropy = calculate_entropy(corpus_text)
    print(f"Entropie pour {corpus_name} : {entropy:.2f}")

    return corpus_sizes, vocab_sizes, k, beta, compression_ratio, entropy


# Sauvegarder les résultats dans un fichier JSON
def save_results_to_json(file_path, results):
    with open(file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)


# Charger les résultats depuis un fichier JSON
def load_results_from_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    return {}


# Initialiser une figure pour afficher plusieurs corpus
plt.figure(figsize=(10, 6))

# Fichier pour sauvegarder les résultats
results_file = "/Users/ghalia/Desktop/Telecom_IA/Algorithmic_Info/algo_info_text_comparison/corpus_analysis_results.json"
results = load_results_from_json(results_file)

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

compression_ratios = {}
entropies = {}

for corpus_name, corpus_text in corpora.items():
    # Si les résultats existent déjà, les charger
    if corpus_name in results:
        corpus_sizes = results[corpus_name]["corpus_sizes"]
        vocab_sizes = results[corpus_name]["vocab_sizes"]
        k = results[corpus_name]["k"]
        beta = results[corpus_name]["beta"]
        compression_ratio = results[corpus_name]["compression_ratio"]
        entropy = results[corpus_name].get("entropy", None)
        print(f"\n--- Résultats chargés pour {corpus_name} ---")
    else:
        # Sinon, calculer les résultats
        corpus_sizes, vocab_sizes, k, beta, compression_ratio, entropy = analyze_corpus(corpus_name, corpus_text)
        # Sauvegarder les résultats
        results[corpus_name] = {
            "corpus_sizes": corpus_sizes,
            "vocab_sizes": vocab_sizes,
            "k": k,
            "beta": beta,
            "compression_ratio": compression_ratio,
            "entropy": entropy
        }

    # Stocker les métriques pour l'analyse globale
    compression_ratios[corpus_name] = compression_ratio
    entropies[corpus_name] = entropy

    # Tracer les données réelles
    plt.loglog(corpus_sizes, vocab_sizes, marker='o', linestyle='-', label=f'{corpus_name} (Données réelles)')
    
    # Tracer la courbe ajustée
    fitted_vocab_sizes = [heaps_law(x, k, beta) for x in corpus_sizes]
    plt.loglog(corpus_sizes, fitted_vocab_sizes, linestyle='--', label=f'{corpus_name} (Ajustement: K={k:.2f}, β={beta:.2f})')

# Sauvegarder les résultats au fichier JSON
save_results_to_json(results_file, results)

# Finaliser le graphique
plt.xlabel('Taille du corpus (log)')
plt.ylabel('Taille du vocabulaire (log)')
plt.title('Comparaison de la loi de Heaps entre différents corpus')
plt.legend()
plt.grid(True)
plt.show()

# Afficher les ratios de compression et l'entropie pour chaque corpus
print("\n--- Résumé des métriques ---")
for corpus_name in corpora.keys():
    print(f"{corpus_name} : K = {results[corpus_name]['k']:.2f}, beta = {results[corpus_name]['beta']:.2f}, Ratio de compression = {compression_ratios[corpus_name]:.2f}, Entropie = {entropies[corpus_name]:.2f}")


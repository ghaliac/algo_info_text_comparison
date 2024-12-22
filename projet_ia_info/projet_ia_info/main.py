import nltk
import requests
from bs4 import BeautifulSoup
from nltk.corpus import gutenberg, brown, reuters, stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import ssl
import numpy as np
import string
from scipy.optimize import curve_fit
import gzip
import json
import os
from math import log2

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
    nltk.download('stopwords')
except Exception as e:
    print(f"Erreur lors du téléchargement des ressources NLTK : {e}")

# Fonctions utilitaires
def extract_vocabulary_size(corpus):
    words = nltk.word_tokenize(corpus)
    fdist = FreqDist(words)
    return len(fdist)

# def clean_corpus(corpus):
#     """
#     Nettoie le corpus en supprimant la ponctuation et en normalisant les espaces.
#     """
#     # Retirer la ponctuation
#     translator = str.maketrans('', '', string.punctuation)
#     cleaned_corpus = corpus.translate(translator)
    
#     # Supprimer les espaces supplémentaires
#     cleaned_corpus = ' '.join(cleaned_corpus.split())
    
#     return cleaned_corpus

def clean_corpus(corpus, language='english'):
    """
    Nettoie le corpus en supprimant la ponctuation, les stop words et en normalisant les espaces.
    """
    # Retirer la ponctuation
    translator = str.maketrans('', '', string.punctuation)
    cleaned_corpus = corpus.translate(translator)
    
    # Supprimer les espaces supplémentaires
    cleaned_corpus = ' '.join(cleaned_corpus.split())
    
    # Tokenisation pour gérer les mots
    words = nltk.word_tokenize(cleaned_corpus)
    
    # Supprimer les stop words
    stop_words = set(stopwords.words(language))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    # Recomposer le texte
    return ' '.join(filtered_words)

def heaps_law(x, k, beta):
    return k * x**beta

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

def save_results_to_json(file_path, results):
    with open(file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

def load_results_from_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    return {}

# Analyse de corpus
def analyze_corpus(corpus_name, corpus_text, step_size=100000):
    print(f"\n--- Analyse du corpus : {corpus_name} ---")
    corpus_sizes = []
    vocab_sizes = []

    for i in range(0, len(corpus_text), step_size):
        subset = corpus_text[:i + step_size]
        corpus_size = len(subset)
        vocab_size = extract_vocabulary_size(subset)
        corpus_sizes.append(corpus_size)
        vocab_sizes.append(vocab_size)
        print(f"Taille du corpus : {corpus_size}, Taille du vocabulaire : {vocab_size}")

    log_corpus_sizes = np.log(corpus_sizes)
    log_vocab_sizes = np.log(vocab_sizes)

    def linear_fit(x, a, b):
        return a + b * x

    params, _ = curve_fit(linear_fit, log_corpus_sizes, log_vocab_sizes)
    log_k, beta = params
    k = np.exp(log_k)

    print(f"Paramètres estimés (log-log) pour {corpus_name} : K = {k:.2f}, β = {beta:.2f}")

    compression_ratio = measure_compression_ratio(corpus_text)
    print(f"Ratio de compression pour {corpus_name} : {compression_ratio:.2f}")

    entropy = calculate_entropy(corpus_text)
    print(f"Entropie pour {corpus_name} : {entropy:.2f}")

    return corpus_sizes, vocab_sizes, k, beta, compression_ratio, entropy

def analyze_zipf(corpus_name, corpus_text):
    print(f"\n--- Analyse de la loi de Zipf pour : {corpus_name} ---")
    words = nltk.word_tokenize(corpus_text)
    fdist = FreqDist(words)
    sorted_frequencies = sorted(fdist.values(), reverse=True)
    ranks = np.arange(1, len(sorted_frequencies) + 1)

    def inverse_fit(x, k):
        return k / x

    params, _ = curve_fit(inverse_fit, ranks, sorted_frequencies)
    k = params[0]
    print(f"Paramètre estimé pour {corpus_name} : k = {k:.2f}")

    plt.figure(figsize=(8, 5))
    max_rank = 100
    plt.scatter(ranks[:max_rank], sorted_frequencies[:max_rank], alpha=0.6, label="Données réelles")
    plt.plot(ranks[:max_rank], inverse_fit(ranks[:max_rank], *params), color='red', label=f"Ajustement : k = {k:.2f}")
    plt.xlabel("Rang")
    plt.ylabel("Fréquence")
    plt.title(f"Loi de Zipf (linéaire) pour {corpus_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return k

# Ajouter les mots des 5 premiers rangs dans un encadré
def add_top_words_box(ax, corpus_name, corpus_text, top_n=10):
    words = nltk.word_tokenize(corpus_text)
    fdist = FreqDist(words)
    most_common_words = fdist.most_common(top_n)  # Les N mots les plus fréquents

    # Préparer le texte à afficher
    text = f"Top {top_n} mots pour {corpus_name}:\n" + "\n".join([f"{rank+1}. {word} ({freq})" for rank, (word, freq) in enumerate(most_common_words)])
    
    # Ajouter une boîte de texte sur le graphique
    ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))


# Fonction pour récupérer et nettoyer le texte des pages web (utilisée dans l'ajout)
def get_web_text(url):
    """
    Récupère et nettoie le texte d'une page web.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        print(f"Erreur lors de l'accès à {url}: {e}")
        return ""

# Initialiser une figure pour afficher plusieurs corpus
plt.figure(figsize=(10, 6))

# Fichier pour sauvegarder les résultats
results_file = "/Users/ghalia/Desktop/Telecom_IA/Algorithmic_Info/algo_info_text_comparison/corpus_analysis_results.json"
results = load_results_from_json(results_file)

# Analyser et tracer chaque corpus
corpora = {
    "Gutenberg (Littéraire)": clean_corpus(''.join([gutenberg.raw(text) for text in [
        'austen-emma.txt', 'shakespeare-hamlet.txt', 'melville-moby_dick.txt',
        'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt',
        'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt'
    ]])),
    "Brown (Généraliste/Conversationnel)": clean_corpus(' '.join(brown.words())),
    "Reuters (Journalistique)": clean_corpus(' '.join(reuters.words()))
}

compression_ratios = {}
entropies = {}
zipf_constants = {}

for corpus_name, corpus_text in corpora.items():
    if corpus_name in results:
        corpus_sizes = results[corpus_name]["corpus_sizes"]
        vocab_sizes = results[corpus_name]["vocab_sizes"]
        k = results[corpus_name]["k"]
        beta = results[corpus_name]["beta"]
        compression_ratio = results[corpus_name]["compression_ratio"]
        entropy = results[corpus_name].get("entropy", None)
        print(f"\n--- Résultats chargés pour {corpus_name} ---")
    else:
        corpus_sizes, vocab_sizes, k, beta, compression_ratio, entropy = analyze_corpus(corpus_name, corpus_text)
        results[corpus_name] = {
            "corpus_sizes": corpus_sizes,
            "vocab_sizes": vocab_sizes,
            "k": k,
            "beta": beta,
            "compression_ratio": compression_ratio,
            "entropy": entropy
        }

    compression_ratios[corpus_name] = compression_ratio
    entropies[corpus_name] = entropy

    # Initialiser le graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Tracer les données réelles
    ax.plot(corpus_sizes, vocab_sizes, marker='o', linestyle='-', label=f'{corpus_name} (Données réelles)')
    fitted_vocab_sizes = [heaps_law(x, k, beta) for x in corpus_sizes]
    ax.plot(corpus_sizes, fitted_vocab_sizes, linestyle='--', label=f'{corpus_name} (Ajustement: K={k:.2f}, β={beta:.2f})')

    # Ajouter les mots des 5 premiers rangs dans une boîte
    add_top_words_box(ax, corpus_name, corpus_text, top_n=5)

    # Ajouter la légende, les grilles, et les titres
    ax.legend()
    ax.grid(True)
    ax.set_title(f'Loi de Heaps-Herden pour {corpus_name}')
    ax.set_xlabel('Taille du corpus')
    ax.set_ylabel('Taille du vocabulaire')

    plt.show()

    zipf_constants[corpus_name] = analyze_zipf(corpus_name, corpus_text)

save_results_to_json(results_file, results)

print("\n--- Résumé des métriques ---")
for corpus_name in corpora.keys():
    print(f"{corpus_name} : K = {results[corpus_name]['k']:.2f}, beta = {results[corpus_name]['beta']:.2f}, Ratio de compression = {compression_ratios[corpus_name]:.2f}, Entropie = {entropies[corpus_name]:.2f}")

print("\n--- Constantes de Zipf (k) pour chaque corpus ---")
for corpus_name, k in zipf_constants.items():
    print(f"{corpus_name} : k = {k:.2f}")


    # Liste des pages web à analyser
urls = [
    "https://en.wikipedia.org/wiki/Natural_language_processing",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning"
]



# Ajout de l’analyse des contenus web
for url in urls:
    print(f"\nTraitement de l'URL : {url}")

    # Extraire le texte de la page web
    web_text = get_web_text(url)

    # Vérifier si le texte est suffisamment long
    if len(web_text) > 100:  # Vérifie si le texte est suffisant pour l’analyse
        # Analyse de la loi de Heaps
        corpus_sizes, vocab_sizes, k, beta, compression_ratio, entropy = analyze_corpus(url, web_text, step_size=1000)

        print(f"\n--- Résultats pour {url} ---")
        print(f"K (Heaps) : {k:.2f}, β : {beta:.2f}")
        print(f"Ratio de compression : {compression_ratio:.2f}")
        print(f"Entropie : {entropy:.2f}")

        # Afficher la courbe de Heaps
        plt.figure(figsize=(10, 6))
        plt.plot(corpus_sizes, vocab_sizes, marker='o', linestyle='-', label='Données réelles')
        fitted_vocab_sizes = [heaps_law(x, k, beta) for x in corpus_sizes]
        plt.plot(corpus_sizes, fitted_vocab_sizes, linestyle='--', label=f'Ajustement : K={k:.2f}, β={beta:.2f}')
        plt.xlabel('Taille du corpus')
        plt.ylabel('Taille du vocabulaire')
        plt.title(f"Loi de Heaps pour {url}")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Analyse de la loi de Zipf
        k_zipf = analyze_zipf(url, web_text)
        print(f"Paramètre k (Zipf) : {k_zipf:.2f}")
    else:
        print(f"Texte insuffisant pour {url}")

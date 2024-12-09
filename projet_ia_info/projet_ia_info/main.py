import zlib
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from transformers import pipeline


# 1. Charger un texte réel
def load_victor_hugo_text():
    """
    Charger un extrait de Victor Hugo.
    """
    return """Il était tombé sur cet homme comme un coup de vent. Il l'avait accablé de son feu. Jean Valjean, éperdu, s'était laissé faire. Il était subjugué. Tantôt il tâchait de se défendre, tantôt il s'humiliait ; mais toujours il résistait avec cette douceur dont il n'avait pas conscience, et il se sentait écrasé par cette bienveillance sévère et douce. La lumière du monseigneur devenait terrible pour lui."""


# 2. Générer un texte dans le style de Victor Hugo
def generate_text_gpt(prompt="Écrire un paragraphe dans le style de Victor Hugo", max_length=200):
    """
    Génère un texte dans le style de Victor Hugo avec un modèle GPT.
    """
    generator = pipeline("text-generation", model="gpt2")  # Remplacez par un modèle plus avancé si possible
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]["generated_text"]


# 3. Calculer la complexité compressée (Kolmogorov)
def kolmogorov_complexity(text):
    """
    Approximation de la complexité de Kolmogorov via la taille compressée.
    """
    compressed = zlib.compress(text.encode('utf-8'))
    return len(compressed)


# 4. Calculer l'entropie de Shannon
def calculate_entropy(text):
    """
    Calcule l'entropie de Shannon en fonction des fréquences des caractères.
    """
    total = len(text)
    frequency = Counter(text)
    probabilities = [freq / total for freq in frequency.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy


# 5. Analyse de la loi de Zipf
def analyze_zipf_law(text, title):
    """
    Analyse la distribution des mots selon la loi de Zipf.
    """
    words = text.lower().split()
    word_counts = Counter(words)
    frequencies = sorted(word_counts.values(), reverse=True)

    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, len(frequencies) + 1), frequencies, marker="o", label="Fréquence des mots")
    plt.title(f"Loi de Zipf : {title}")
    plt.xlabel("Rang (log)")
    plt.ylabel("Fréquence (log)")
    plt.grid()
    plt.legend()
    plt.show()


# 6. Comparaison des textes
if __name__ == "__main__":
    # Charger les textes
    print("Génération des textes...")
    victor_hugo_text = load_victor_hugo_text()
    generated_text = generate_text_gpt(prompt="Écrire un paragraphe dans le style de Victor Hugo", max_length=200)

    # Afficher les textes
    print("=== Texte de Victor Hugo ===")
    print(victor_hugo_text)
    print("\n=== Texte Généré ===")
    print(generated_text)

    # Complexité compressée
    print("\nCalcul de la complexité compressée...")
    complexity_hugo = kolmogorov_complexity(victor_hugo_text)
    complexity_generated = kolmogorov_complexity(generated_text)
    print(f"Complexité compressée (Victor Hugo) : {complexity_hugo} octets")
    print(f"Complexité compressée (GPT) : {complexity_generated} octets")

    # Entropie
    print("\nCalcul de l'entropie de Shannon...")
    entropy_hugo = calculate_entropy(victor_hugo_text)
    entropy_generated = calculate_entropy(generated_text)
    print(f"Entropie (Victor Hugo) : {entropy_hugo:.2f} bits/caractère")
    print(f"Entropie (GPT) : {entropy_generated:.2f} bits/caractère")

    # Loi de Zipf
    print("\nAnalyse de la loi de Zipf...")
    print("Victor Hugo :")
    analyze_zipf_law(victor_hugo_text, title="Victor Hugo")
    print("Texte Généré :")
    analyze_zipf_law(generated_text, title="Texte Généré")

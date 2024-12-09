import zlib
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from transformers import pipeline


# 1. Charger un texte réel
def load_victor_hugo_text():
    """
    Charger un extrait de Shakespeare.
    """
    return """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them? To die: to sleep;
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to, 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep: perchance to dream: ay, there's the rub;
    For in that sleep of death what dreams may come
    When we have shuffled off this mortal coil,
    Must give us pause: there's the respect
    That makes calamity of so long life;
    For who would bear the whips and scorns of time,
    The oppressor's wrong, the proud man's contumely,
    The pangs of despised love, the law's delay,
    The insolence of office and the spurns
    That patient merit of the unworthy takes,
    When he himself might his quietus make
    With a bare bodkin? who would fardels bear,
    To grunt and sweat under a weary life,
    But that the dread of something after death,
    The undiscovered country from whose bourn
    """
    # No traveller returns, puzzles the will
    # And makes us rather bear those ills we have
    # Than fly to others that we know not of?
    # Thus conscience does make cowards of us all;
    # And thus the native hue of resolution
    # Is sicklied o'er with the pale cast of thought,
    # And enterprises of great pitch and moment
    # With this regard their currents turn awry,
    # And lose the name of action.
    #""""


# 2. Générer un texte dans le style de Shakespeare
def generate_text_gpt(prompt="Generates a text in the style of Shakespeare", max_length=300):

    generator = pipeline("text-generation", model="gpt2")  # Replace with a more advanced model if possible
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


# Analyse de la loi de Zipf pour deux textes
def analyze_zipf_law_comparison(text1, text2, title1="Texte 1", title2="Texte 2"):
    """
    Compare la distribution des mots selon la loi de Zipf entre deux textes et les affiche côte à côte.
    """
    # Préparer les données pour le premier texte
    words1 = text1.lower().split()
    word_counts1 = Counter(words1)
    frequencies1 = sorted(word_counts1.values(), reverse=True)

    # Préparer les données pour le second texte
    words2 = text2.lower().split()
    word_counts2 = Counter(words2)
    frequencies2 = sorted(word_counts2.values(), reverse=True)

    # Création des sous-graphiques
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Graphe pour le premier texte
    axes[0].loglog(range(1, len(frequencies1) + 1), frequencies1, marker="o", label=title1)
    axes[0].set_title(f"Loi de Zipf : {title1}")
    axes[0].set_xlabel("Rang (log)")
    axes[0].set_ylabel("Fréquence (log)")
    axes[0].grid()
    axes[0].legend()

    # Graphe pour le second texte
    axes[1].loglog(range(1, len(frequencies2) + 1), frequencies2, marker="o", label=title2)
    axes[1].set_title(f"Loi de Zipf : {title2}")
    axes[1].set_xlabel("Rang (log)")
    axes[1].grid()
    axes[1].legend()

    # Afficher les deux graphiques
    plt.tight_layout()
    plt.show()
    plt.show()


def calculate_lexical_richness(text):
    """
    Calcule la richesse lexicale d'un texte.
    Richesse lexicale = Nombre de mots uniques / Nombre total de mots
    """
    words = text.lower().split()
    total_words = len(words)
    unique_words = len(set(words))
    lexical_richness = unique_words / total_words
    return lexical_richness, total_words, unique_words


# 6. Comparaison des textes
if __name__ == "__main__":
    # Charger les textes
    print("Génération des textes...")
    shakespeare_text = load_victor_hugo_text()
    generated_text = generate_text_gpt(prompt="Generate a text in the style of Shakespeare", max_length=300)

    # Afficher les textes
    print("=== Texte de Shakespeare ===")
    print(shakespeare_text)
    print("\n=== Texte Généré ===")
    print(generated_text)

    # Complexité compressée
    print("\nCalcul de la complexité compressée...")
    complexity_shakespeare = kolmogorov_complexity(shakespeare_text)
    complexity_generated = kolmogorov_complexity(generated_text)
    print(f"Complexité compressée (Shakespeare) : {complexity_shakespeare} octets")
    print(f"Complexité compressée (GPT) : {complexity_generated} octets")

    # Entropie
    print("\nCalcul de l'entropie de Shannon...")
    entropy_shakespeare = calculate_entropy(shakespeare_text)
    entropy_generated = calculate_entropy(generated_text)
    print(f"Entropie (Shakespeare) : {entropy_shakespeare:.2f} bits/caractère")
    print(f"Entropie (GPT) : {entropy_generated:.2f} bits/caractère")

    # Richesse lexicale
    print("\nCalcul de la richesse lexicale...")
    richness_shakespeare, total_words_shakespeare, unique_words_shakespeare = calculate_lexical_richness(shakespeare_text)
    richness_generated, total_words_generated, unique_words_generated = calculate_lexical_richness(generated_text)
    print(f"Richesse lexicale (Shakespeare) : {richness_shakespeare:.2f} (Unique: {unique_words_shakespeare}, Total: {total_words_shakespeare})")
    print(f"Richesse lexicale (GPT) : {richness_generated:.2f} (Unique: {unique_words_generated}, Total: {total_words_generated})")

    # Loi de Zipf
    print("\nAnalyse de la loi de Zipf...")
    analyze_zipf_law_comparison(
        text1=shakespeare_text,
        text2=generated_text,
        title1="Shakespeare",
        title2="GPT text in the style of Shakespeare"
    )

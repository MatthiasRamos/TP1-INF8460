"""
Questions 1.1.6 à 1.1.8 : calcul de différentes statistiques sur un corpus.

Sauf mention contraire, toutes les fonctions renvoient un nombre (int ou float).
Dans toutes les fonctions de ce fichier, le paramètre `corpus` désigne une liste de phrases tokenizées, par exemple :
>> corpus = [
    ["Alice", "est", "là"],
    ["Bob", "est", "ici"]
]
"""
import preprocess_corpus as pre
from collections import Counter

def count_tokens(corpus):
    """
    Renvoie le nombre de mots dans le corpus
    """
    numberOfTokens = 0
    for sentence in corpus:
        numberOfTokens += len(sentence)
    return numberOfTokens


def count_types(corpus):
    """
    Renvoie le nombre de types (mots distincts) dans le corpus
    """
    words = {}
    for sent in corpus:
        for word in sent:
            words[word] = True
    return len(words)


def get_most_frequent(corpus, n):
    """
    Renvoie les n mots les plus fréquents dans le corpus, ainsi que leurs fréquences

    :return: list(tuple(str, float)), une liste de paires (mot, fréquence)
    """
    words = {}
    for sent in corpus:
        for word in sent:
            words[word] = words.get(word,0) + 1
    return sorted(words.items(),key=lambda x: -x[1])[:n]


def get_token_type_ratio(corpus):
    """
    Renvoie le ratio nombre de tokens sur nombre de types
    """
    return count_tokens(corpus)/count_types(corpus)


def count_lemmas(corpus):
    """
    Renvoie le nombre de lemmes distincts
    """
    lemmatized_corpus = pre.lemmatize(corpus) # TODO: Do we stem before lemmatize ?
    return count_types(lemmatized_corpus)

def count_stems(corpus):
    """
    Renvoie le nombre de racines (stems) distinctes
    """
    stem_corpus = pre.stem(corpus)
    return count_types(stem_corpus)


def explore(corpus):
    """
    Affiche le résultat des différentes fonctions ci-dessus.

    Pour `get_most_frequent`, prenez n=15

    >> explore(corpus)
    Nombre de tokens: 5678
    Nombre de types: 890
    ...
    Nombre de stems: 650

    (Les chiffres ci-dessus sont indicatifs et ne correspondent pas aux résultats attendus)
    """
    print(f"Le nombre total de tokens: {count_tokens(corpus)}")
    print(f"Le nombre total de mots distincts: {count_types(corpus)}")
    print(f"Les 15 mots les plus fréquents du vocabulaire ainsi que leur fréquence: {get_most_frequent(corpus,15)}")
    print(f"Le ratio token/type: {get_token_type_ratio(corpus)}")
    print(f"Le nombre total de lemmes distincts: {count_lemmas(corpus)}")
    print(f"Le nombre total de racines distinctes: {count_stems(corpus)}")

if __name__ == "__main__":
    """
    Ici, appelez la fonction `explore` sur `shakespeare_train` et `shakespeare_test`. Quand on exécute le fichier, on
    doit obtenir :

    >> python explore_corpus
    -- shakespeare_train --
    Nombre de tokens: 5678
    Nombre de types: 890
    ...

    -- shakespeare_test --
    Nombre de tokens: 78009
    Nombre de types: 709
    ...
    """
    print("-- shakespeare_train --")
    corpus = pre.read_and_preprocess("output/shakespeare_train_phrases.txt")
    explore(corpus)

    print("-- shakespeare_test --")
    corpus = pre.read_and_preprocess("output/shakespeare_test_phrases.txt")
    explore(corpus)

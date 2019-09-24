"""
Questions 1.4.1 à 1.6.2 : modèles de langue NLTK

Dans ce fichier, on rassemble les fonctions concernant les modèles de langue NLTK :
- entraînement d'un modèle de langue sur un corpus d'entraînement, avec ou sans lissage
- évaluation d'un modèle sur un corpus de test
- génération de texte suivant un modèle de langue

Pour préparer les données avant d'utiliser un modèle, on pourra utiliser
>>> ngrams, words = padded_everygram_pipeline(n, corpus)
>>> vocab = Vocabulary(words, unk_cutoff=k)

Lors de l'initialisation d'un modèle, il faut passer une variable order qui correspond à l'ordre du modèle n-gramme,
et une variable vocabulary de type `Vocabulary`.

On peut ensuite entraîner le modèle avec la méthode `model.fit(ngrams)`
"""
from nltk.lm.models import MLE, Laplace, Lidstone
from nltk.lm.vocabulary import Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import RegexpTokenizer
from preprocess_corpus import read_and_preprocess
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt

def train_LM_model(corpus, model, n, gamma=None, unk_cutoff=2):
    """
    Entraîne un modèle de langue n-gramme NLTK de la classe `model` sur le corpus.

    :param corpus: list(list(str)), un corpus tokenizé
    :param model: un des éléments de (MLE, Lidstone, Laplace)
    :param n: int, l'ordre du modèle
    :param gamma: float or None, le paramètre gamma (pour `model=Lidstone` uniquement). Si model=Lidstone, alors cet
    argument doit être renseigné
    :param unk_cutoff: le seuil au-dessous duquel un mot est considéré comme inconnu et remplacé par <UNK>
    :return: un modèle entraîné
    """
    if model not in [MLE, Laplace, Lidstone]:
        raise TypeError("Unkown model type! supported types: (MLE, Lidstone, Laplace)")

    ngrams, words = padded_everygram_pipeline(n, corpus)
    vocab = Vocabulary(words, unk_cutoff=unk_cutoff)

    params = {
        "order":n,
        "vocabulary":vocab,
    }
    if model == Lidstone:
        params["gamma"] = gamma
    ist_model = model(**params)
    ist_model.fit(ngrams)
    
    return ist_model

def evaluate(model, corpus):
    """
    Renvoie la perplexité du modèle sur une phrase de test.

    :param model: nltk.lm.api.LanguageModel, un modèle de langue
    :param corpus: list(list(str)), une corpus tokenizé
    :return: float
    """
    return model.perplexity(corpus)


def evaluate_gamma(gamma, train, test, n):
    """
    Entraîne un modèle Lidstone n-gramme de paramètre `gamma` sur un corpus `train`, puis retourne sa perplexité sur un
    corpus `test`.

    [Cette fonction est déjà complète, on ne vous demande pas de l'écrire]

    :param gamma: float, la valeur de gamma (comprise entre 0 et 1)
    :param train: list(list(str)), un corpus d'entraînement
    :param test: list(list(str)), un corpus de test
    :param n: l'ordre du modèle
    :return: float, la perplexité du modèle sur train
    """
    lm = train_LM_model(train, Lidstone, n, gamma=gamma)
    return evaluate(lm, test)


def generate(model, n_words, text_seed=None, random_seed=None):
    """
    Génère `n_words` mots à partir du modèle.

    Vous utiliserez la méthode `model.generate(num_words, text_seed, random_seed)` de NLTK qui permet de générer
    du texte suivant la distribution de probabilité du modèle de langue.

    Cette fonction doit renvoyer une chaîne de caractère détokenizée (dans le cas de Trump, vérifiez que les # et les @
    sont gérés); si le modèle génère un symbole de fin de phrase avant d'avoir fini, vous devez recommencer une nouvelle
    phrase, jusqu'à avoir produit `n_words`.

    :param model: un modèle de langue entraîné
    :param n_words: int, nombre de mots à générer
    :param text_seed: tuple(str), le contexte initial. Si aucun text_seed n'est précisé, vous devrez utiliser le début
    d'une phrase, c'est à dire respectivement (), ("<s>",) ou ("<s>", "<s>") pour n=1, 2, 3
    :param random_seed: int, la seed à passer à la méthode `model.generate` pour pouvoir reproduire les résultats. Pour
    ne pas fixer de seed, il suffit de laisser `random_seed=None`
    :return: str
    """
    #control randommization for constant generation of tokens
    np.random.seed(random_seed)
    n = model.order
    if text_seed is None:
        text_seed = ["<s>"]*(n-1)
    if len(text_seed)!=n-1:
        raise ValueError(f"Inconsistency in the size of text_seed, got {len(text_seed)}, expected {n}")

    text_words = []
    regenerate = True
    while regenerate:
        tokens_generated = list(model.generate(n_words, text_seed, random_seed))
        regenerate = False
        if "</s>" in tokens_generated:
            i = tokens_generated.index("</s>")
            if i < n_words - model.order:
                # remaining number of words to generate next!
                n_words -= i
                # keep the first sentence!
                tokens_generated = tokens_generated[:i]
                # change the seed to have a new begining for the new sentence!
                if not random_seed is None:
                    random_seed = np.random.randint(1000)
                # keep looping
                regenerate = True
        text_words+=tokens_generated
    return " ".join(text_words)


if __name__ == "__main__":
    print("Loading data...")
    corpus = read_and_preprocess("output/shakespeare_train_lemmes.txt")
    test = read_and_preprocess("output/shakespeare_test_lemmes.txt")
    """
    Vous aurez ici trois tâches à accomplir ici :
    
    1)
    Dans un premier temps, vous devez entraîner des modèles de langue MLE et Laplace pour n=1, 2, 3 à l'aide de la 
    fonction `train_MLE_model` sur le corpus `shakespeare_train` (question 1.4.2). Puis vous devrez évaluer vos modèles 
    en mesurant leur perplexité sur le corpus `shakespeare_test` (question 1.5.2).
    """
    print("-"*40)
    print("Q1")
    print("-"*40)
    for n in [1,2,3]:
        print(f"n={n}")
        print("-"*20)
        print(f"[+] fitting models with n={n}")
        trained_mle = train_LM_model(corpus, MLE, n)
        trained_laplace = train_LM_model(corpus, Laplace, n)

        print(f"perplexity mle= {evaluate(trained_mle,test)}")
        print(f"perplexity laplace= {evaluate(trained_laplace,test)}")
    """
    >output:
    n=1
    --------------------
    [+] fitting models with n=1
    perplexity mle= inf
    perplexity laplace= 14716.000000000042
    n=2
    --------------------
    [+] fitting models with n=2
    perplexity mle= inf
    perplexity laplace= 11782.372307261669
    n=3
    --------------------
    [+] fitting models with n=3
    perplexity mle= inf
    perplexity laplace= 11377.463113703267

    2)
    Ensuite, on vous demande de tracer un graphe représentant le perplexité d'un modèle Lidstone en fonction du paramètre 
    gamma. Vous pourrez appeler la fonction `evaluate_gamma` (déjà écrite) sur `shakespeare_train` et `shakespeare_test` 
    en faisant varier gamma dans l'intervalle (10^-5, 1) (question 1.5.3). Vous utiliserez une échelle logarithmique en 
    abscisse et en ordonnée.
    
    Note : pour les valeurs de gamma à tester, vous pouvez utiliser la fonction `numpy.logspace(-5, 0, 10)` qui renvoie 
    une liste de 10 nombres, répartis logarithmiquement entre 10^-5 et 1.
    """
    print("-"*40)
    print("Q2")
    print("-"*40)
    os.makedirs("output",exist_ok=True)
    results = defaultdict(list)
    print("Training models")
    print("-"*20)
    for n in range(1,4):
        plt.figure()
        print(f"  models with n={n}",end="")
        for gamma in np.logspace(-5, 0, 10):
            results[n].append(evaluate_gamma(gamma,corpus,test,n))
            print(".",end="")
        print("")
        with open(os.path.join("output",f"Lipston_{n}.csv"),"w") as fout:
            fout.write("\n".join(list(map(lambda x:f"{x[0]},{x[1]}",zip(np.logspace(-5, 0, 10), results[n])))))
        plt.plot(np.logspace(-5, 0, 10), results[n])
        plt.xlabel("gamma")
        plt.ylabel("perplexity value")
        plt.title(f"Evolution of perplexity with respect to different values of gamma for n = {n}")
        plt.savefig(os.path.join("output",f"Lipston_{n}.png"))
    """
    3)
    Enfin, pour chaque n=1, 2, 3, vous devrez générer 2 segments de 20 mots pour des modèles MLE entraînés sur Trump.
    Réglez `unk_cutoff=1` pour éviter que le modèle ne génère des tokens <UNK> (question 1.6.2).
    """
    print("-"*40)
    print("Q3")
    print("-"*40)
    tokenizer = RegexpTokenizer(r'([@#]?\w+|\S)')
    with open("data/trump.txt","r") as fin:
        # read raw data
        raw_data = fin.readlines()
        # tockenize
        corpus_trump = list(map(lambda x:tokenizer.tokenize(x.replace("&amp;","&")),raw_data))

    for n in [1,2,3]:
        print(f"n={n}")
        print("-"*20)
        print(f"[+] fitting model MLE with n={n}")
        trained_mle = train_LM_model(corpus_trump, MLE, n, unk_cutoff=1)
        print("Generated text:")
        print("-"*20)
        for s in range(2):
            generated = generate(trained_mle, n_words=20, random_seed=1002+s) 
            print(f"Segment {s+1}:\n",generated)


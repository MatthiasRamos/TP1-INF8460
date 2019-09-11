"""
Questions 1.1.1 à 1.1.5 : prétraitement des données.
"""
import nltk
import matplotlib
import numpy

nltk.download("punkt")
nltk.download("wordnet")


def segmentize(raw_text):
    """
    Segmente un texte en phrases.

    >> raw_corpus = "Alice est là. Bob est ici"
    >> segmentize(raw_corpus)
    ["Alice est là.", "Bob est ici"]

    :param raw_text: str
    :return: list(str)
    """
    return nltk.sent_tokenize(raw_text)



def tokenize(sentences):
    """
    Tokenize une liste de phrases en mots.

    >> sentences = ["Alice est là", "Bob est ici"]
    >> corpus = tokenize(sentences)
    >> corpus
    [
        ["Alice", "est", "là"],
        ["Bob", "est", "ici"]
    ]

    :param sentences: list(str), une liste de phrases
    :return: list(list(str)), une liste de phrases tokenizées
    """
    return [nltk.word_tokenize(phrase) for phrase in sentences]




def lemmatize(corpus):
    """
    Lemmatise les mots d'un corpus.

    :param corpus: list(list(str)), une liste de phrases tokenizées
    :return: list(list(str)), une liste de phrases lemmatisées
    """
    lemmzer = nltk.WordNetLemmatizer()
    lems = []
    for sent in corpus:
        lems.append([lemmzer.lemmatize(token) for token in sent])
    return lems

def stem(corpus):
    """
    Retourne la racine (stem) des mots d'un corpus.

    :param corpus: list(list(str)), une liste de phrases tokenizées
    :return: list(list(str)), une liste de phrases stemées
    """
    stemmer = nltk.PorterStemmer()
    stems = []
    for sent in corpus:
        stems.append([stemmer.stem(token) for token in sent])
    return stems

def read_and_preprocess(filename):
    """
    Lit un fichier texte, puis lui applique une segmentation et une tokenization.

    [Cette fonction est déjà complète, on ne vous demande pas de l'écrire]

    :param filename: str, nom du fichier à lire
    :return: list(list(str))
    """
    with open(filename, "r") as f:
        raw_text = f.read()
    return tokenize(segmentize(raw_text))


def test_preprocessing(raw_text, sentence_id=0):
    """
    Applique à `raw_text` les fonctions segmentize, tokenize, lemmatize et stem, puis affiche le résultat de chacune
    de ces fonctions à la phrase d'indice `sentence_id`

    >> trump = open("data/trump.txt", "r").read()
    >> test_preprocessing(trump)
    Today we express our deepest gratitude to all those who have served in our armed forces.
    Today,we,express,our,deepest,gratitude,to,all,those,who,have,served,in,our,armed,forces,.
    Today,we,express,our,deepest,gratitude,to,all,those,who,have,served,in,our,armed,force,.
    today,we,express,our,deepest,gratitud,to,all,those,who,have,serv,in,our,arm,forc,.

    :param raw_text: str, un texte à traiter
    :param sentence_id: int, l'indice de la phrase à afficher
    :return: un tuple (sentences, tokens, lemmas, stems) qui contient le résultat des quatre fonctions appliquées à
    tout le corpus
    """




if __name__ == "__main__":
    """
    Appliquez la fonction `test_preprocessing` aux corpus `shakespeare_train` et `shakespeare_test`.

    Note : ce bloc de code ne sera exécuté que si vous lancez le script directement avec la commande :
    ```
    python preprocess_corpus.py
    ```
    """

    raw_text_test = open("data/shakespeare_test.txt", "r").read()
    raw_text_train = open("data/shakespeare_train.txt", "r").read()
    with open("output/shakespeare_test_phrases.txt", "w") as f:
        for sentence in segmentize(raw_text_test):
            f.write(sentence + " \n")
    with open("output/shakespeare_train_phrases.txt", "w") as f:
        for sentence in segmentize(raw_text_train):
            f.write(sentence + " \n")
    with open("output/shakespeare_test_mots.txt", "w") as f:
        for sentence in tokenize(segmentize(raw_text_test)):
            for word in sentence:
                f.write(word + " ")
            f.write("\n")
    with open("output/shakespeare_train_mots.txt", "w") as f:
        for sentence in tokenize(segmentize(raw_text_train)):
            for word in sentence:
                f.write(word + " ")
            f.write("\n")

#    open("output/shakespeare_train_phrases.txt").write(segmentize(raw_text_train))
 #   open("output/shakespeare_test_mots.txt").write(tokenize(segmentize(raw_text_test)))
  #  open("output/shakespeare_train_mots.txt").write(tokenize(segmentize(raw_text_train)))



    print(segmentize("Alice est là. Bob est ici"))

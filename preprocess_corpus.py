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
    lm = nltk.WordNetLemmatizer()
    return [list(map(lm.lemmatize, st)) for st in corpus]

def stem(corpus):
    """
    Retourne la racine (stem) des mots d'un corpus.

    :param corpus: list(list(str)), une liste de phrases tokenizées
    :return: list(list(str)), une liste de phrases stemées
    """
    stemmer = nltk.PorterStemmer()
    return [list(map(stemmer.stem, st)) for st in corpus]

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
    sg_text = segmentize(raw_text)
    print(sg_text[sentence_id])

    tk_text = tokenize(sg_text)
    print(tk_text[sentence_id])

    lm_text = lemmatize(tk_text)
    print(lm_text[sentence_id])

    stem_text = stem(tk_text)
    print(stem_text[sentence_id])

    return sg_text, tk_text, lm_text, stem_text


if __name__ == "__main__":
    """
    Appliquez la fonction `test_preprocessing` aux corpus `shakespeare_train` et `shakespeare_test`.

    Note : ce bloc de code ne sera exécuté que si vous lancez le script directement avec la commande :
    ```
    python preprocess_corpus.py
    ```
    """
    import os
    import re
    data_folder = "data"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(data_folder, "shakespeare_test.txt"), "r") as f:
        raw_text = f.read().split("\n\n",1)[1]
        raw_text = " ".join(re.split("\n\n\w+\n\n", raw_text)).replace("\n"," ")

        sg_text, tk_text, lm_text, stem_text = test_preprocessing(raw_text)

        # save sentences
        with open(os.path.join(output_folder, "shakespeare_test_phrases.txt"),"w") as fout:
            fout.write("\n".join(sg_text))

        # save words
        with open(os.path.join(output_folder, "shakespeare_test_mots.txt"),"w") as fout:
            fout.write("\n".join(map(lambda list_words: " ".join(list_words),tk_text)))

        # save lemmes
        with open(os.path.join(output_folder, "shakespeare_test_lemmes.txt"),"w") as fout:
            fout.write("\n".join(map(lambda list_words: " ".join(list_words),lm_text)))
        
        # save stems
        with open(os.path.join(output_folder, "shakespeare_test_stems.txt"),"w") as fout:
            fout.write("\n".join(map(lambda list_words: " ".join(list_words),stem_text)))

    with open(os.path.join(data_folder, "shakespeare_train.txt"), "r") as f:
        raw_text = f.read().split("\n",1)[1]
        raw_text = " ".join(re.split("\n\n.*:", raw_text)).replace("\n"," ")

        sg_text, tk_text, lm_text, stem_text = test_preprocessing(raw_text)

        # save sentences
        with open(os.path.join(output_folder, "shakespeare_train_phrases.txt"),"w") as fout:
            fout.write("\n".join(sg_text))

        # save words
        with open(os.path.join(output_folder, "shakespeare_train_mots.txt"),"w") as fout:
            fout.write("\n".join(map(lambda list_words: " ".join(list_words),tk_text)))

        # save lemmes
        with open(os.path.join(output_folder, "shakespeare_train_lemmes.txt"),"w") as fout:
            fout.write("\n".join(map(lambda list_words: " ".join(list_words),lm_text)))
        
        # save stems
        with open(os.path.join(output_folder, "shakespeare_train_stems.txt"),"w") as fout:
            fout.write("\n".join(map(lambda list_words: " ".join(list_words),stem_text)))
    
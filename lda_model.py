from gensim import corpora, models

def train_lda(texts):
    tokenized = [t.split() for t in texts]
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(t) for t in tokenized]

    lda = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=3,
        passes=10
    )

    return lda.print_topics()

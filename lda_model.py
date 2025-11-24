from gensim import corpora, models

def train_lda(texts):
    tokenized = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(t) for t in tokenized]

    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=3,
        passes=10
    )

    topics = lda_model.print_topics()
    return topics

from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel

def train_lda(texts):
    tokenized = [t.split() for t in texts]
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]

    lda = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=3,
        random_state=42
    )

    topics = lda.print_topics()

    coherence = CoherenceModel(
        model=lda,
        texts=tokenized,
        dictionary=dictionary,
        coherence='c_v'
    ).get_coherence()

    return topics, coherence

from bertopic import BERTopic

# BERTopic requires at least 2–5 documents to train
dummy_docs = [
    "technology is advancing fast",
    "artificial intelligence changes human life",
    "internet and smartphones improve communication",
    "digital tools transform education",
    "science and innovation drive progress"
]

model = BERTopic()
model.fit(dummy_docs)

def get_bert_topic(text):
    topic, _ = model.transform([text])
    return topic

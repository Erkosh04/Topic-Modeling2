from bertopic import BERTopic

dummy_docs = [
    "technology is advancing fast",
    "artificial intelligence changes human life",
    "internet and smartphones improve communication",
    "digital tools transform education",
    "science and innovation drive progress"
]

# UMAP-ты өшіреміз → қатесіз жұмыс істейді
model = BERTopic(umap_model=None)

model.fit(dummy_docs)

def get_bert_topic(text):
    topic, _ = model.transform([text])
    return topic

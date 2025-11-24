from bertopic import BERTopic

# Dummy training data (BERTopic requires multiple docs)
dummy_texts = [
    "technology is advancing rapidly in modern society",
    "artificial intelligence impacts human life",
    "internet and smartphones changed communication",
    "science and innovation drive progress",
    "education improves with digital tools"
]

model = BERTopic()
model.fit(dummy_texts)

def get_bert_topic(text):
    topics, _ = model.transform([text])
    return topics

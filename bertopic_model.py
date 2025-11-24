from bertopic import BERTopic

def train_bertopic(texts):
    model = BERTopic()
    topics, _ = model.fit_transform(texts)
    return model.get_topic_info()

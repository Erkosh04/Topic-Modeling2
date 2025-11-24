import streamlit as st
from preprocessing import preprocess_text
from lda_model import train_lda
from bertopic_model import get_bert_topic

st.title("Topic Modeling App (LDA + BERTopic)")

text = st.text_area("Мәтін енгізіңіз:")

model_choice = st.selectbox("Модельді таңдаңыз:", ["LDA", "BERTopic"])

if st.button("Модельді іске қосу"):
    if not text.strip():
        st.error("Мәтін енгізіңіз!")
    else:
        clean = preprocess_text(text)
        st.write("Өңделген мәтін:")
        st.write(clean)

        if model_choice == "LDA":
            topics = train_lda([clean])
            st.write("LDA Тақырыптары:")
            st.write(topics)

        elif model_choice == "BERTopic":
            topic = get_bert_topic(clean)
            st.write("BERTopic Тақырыбы:")
            st.write(topic)

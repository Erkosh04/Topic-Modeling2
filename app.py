import streamlit as st
from preprocessing import preprocess_text
from lda_model import train_lda
from bertopic_model import train_bertopic

st.title("Topic Modeling (LDA & BERTopic)")

text = st.text_area("Мәтін енгізіңіз:")

model = st.selectbox("Модельді таңдаңыз:", ["LDA", "BERTopic"])

if st.button("Жіберу"):
    if len(text) < 5:
        st.error("Мәтін тым қысқа!")
    else:
        clean = preprocess_text(text)
        st.write("Өңделген мәтін:")
        st.write(clean)

        if model == "LDA":
            topics, coh = train_lda([clean])
            st.subheader("LDA Тақырыптары:")
            st.write(topics)
            st.write("Coherence:", coh)

        else:
            topics = train_bertopic([clean])
            st.subheader("BERTopic Тақырыптары:")
            st.write(topics)

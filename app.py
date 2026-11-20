import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Emotion Detector", page_icon="😊")

@st.cache_resource
def load_model():
    return pipeline("text-classification",
                    model="Sammi111/emotion-distilbert",
                    top_k=None)

clf = load_model()

st.title("😊 Emotion Detection from Text")
st.write("Type a sentence and the model will predict the emotion "
         "(sadness, joy, love, anger, fear, surprise).")

text = st.text_area("Your text:", height=120)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        scores = sorted(clf(text)[0], key=lambda x: x["score"], reverse=True)
        top = scores[0]
        st.success(f"Predicted emotion: **{top['label']}** ({top['score']:.1%})")
        st.bar_chart({s["label"]: s["score"] for s in scores})

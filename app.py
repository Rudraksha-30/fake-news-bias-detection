import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Fake News & Bias Detector")

st.title("🧠 AI-Powered Fake News & Bias Detector")

st.write("Enter a news statement to analyze:")

text = st.text_area("News Text")

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):

            classifier = pipeline("sentiment-analysis")
            result = classifier(text)[0]

            label = result["label"]
            confidence = result["score"]

            if label == "NEGATIVE":
                prediction = "Potentially Misleading / Fake"
            else:
                prediction = "Likely Genuine"

            bias_keywords = ["propaganda", "agenda", "corrupt", "fake", "hate"]
            bias_score = sum(word in text.lower() for word in bias_keywords)

            st.subheader("Results")

            st.write("Prediction:", prediction)
            st.write("Confidence:", round(confidence * 100, 2), "%")

            if bias_score > 0:
                st.error("Bias Indicators Detected!")
            else:
                st.success("No strong bias indicators detected.")

            trust_score = confidence * 100

            fig, ax = plt.subplots()
            ax.bar(["Trust Score"], [trust_score])
            ax.set_ylim(0, 100)
            ax.set_ylabel("Score (%)")

            st.pyplot(fig)

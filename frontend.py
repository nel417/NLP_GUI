import streamlit

from main import get_text_characteristics, \
    get_text_sentiment, \
    get_entity, \
    sentiment_from_scratch


def run():

    streamlit.title("Lets Analyze Text")
    sentence = streamlit.text_input("Type Sentence Here")

    streamlit.write("Analyze text")
    if streamlit.button("Analyze Text", key=1):
        analyze = get_text_characteristics(sentence)
        streamlit.success(f"Results: \n {analyze}")

    streamlit.write("Analyze sentiment (spacy)")
    if streamlit.button("Analyze Sentiment", key=2):
        analyze = get_text_sentiment(sentence)
        streamlit.success(f"Results: \n {analyze}")

    streamlit.write("Analyze Named Entities")
    if streamlit.button("Named Entity Recognition", key="3"):
        analyze = get_entity(sentence)
        streamlit.success(f"Results \n {analyze}")

    streamlit.write("Analyze Sentiment (scikit learn SVM)")
    if streamlit.button("Sentiment + F1 Score", key="4"):
        analyze = sentiment_from_scratch(sentence)
        streamlit.success(f"Results \n {analyze}")



run()

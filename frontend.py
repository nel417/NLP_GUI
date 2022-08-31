import streamlit

from main import get_text_characteristics, get_text_sentiment, get_entity


def run():
    streamlit.title("Lets Analyze Text")
    sentence = streamlit.text_input("Type Sentence Here")

    if streamlit.button("Analyze Text", key=1):
        analyze = get_text_characteristics(sentence)
        streamlit.success(f"Results: \n {analyze}")

    if streamlit.button("Analyze Sentiment", key=2):
        analyze = get_text_sentiment(sentence)
        streamlit.success(f"Results: \n {analyze}")

    if streamlit.button("Named Entity Recognition", key="3"):
        analyze = get_entity(sentence)
        streamlit.success(f"Results \n {analyze}")


run()

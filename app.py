import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
model = joblib.load("fake_news_classifier_model.pkl")

# Load the TF-IDF Vectorizer (if needed)
vectorizer = joblib.load("vectorizer.pkl")  # Save and load this if needed

# Streamlit UI
st.title("News Bot")
st.write("Enter a news title to check if it's real or fake.")


# Initialize chat history 
if "messages" not in st.session_state: 
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if prompt:
        # Transform user input using the same TF-IDF vectorizer
        user_input_tfidf = vectorizer.transform([prompt])

        # Make prediction
        prediction = model.predict(user_input_tfidf)[0]
        class_names = ["Fake News", "Real News"]

        # Show result
        response = f"**Prediction:** {class_names[prediction]}"
    else:
        st.warning("Please enter text before predicting.")

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


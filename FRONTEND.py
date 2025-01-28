import streamlit as st
import pandas as pd
import time
from BACKEND import (
    initialize_google_sheets,
    record_audio_chunk,
    transcribe_audio,
    analyze_sentiment,
    extract_laptop_name,
    get_deal_recommendations,
    negotiation_coach,
    get_previous_interaction,
    summarize_conversation,
    find_customer_row,
    append_to_existing_customer_row,
    append_new_customer_row,
)

# Load the laptop dataset
laptop_data = pd.read_csv("C:\\Users\\choud\\Downloads\\laptop_dataset_updated.csv")  # Path to the dataset
laptop_data = laptop_data.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

# Helper functions
def extract_bigrams_from_text(text):
    words = text.lower().split()
    return [" ".join(words[i:i + 2]) for i in range(len(words) - 1)]

def extract_laptop_name(user_input, dataset):
    words = user_input.lower().split()
    bigrams = extract_bigrams_from_text(user_input)

    for bigram in bigrams:
        matching_rows = dataset[dataset["Product Name"].str.contains(bigram, case=False, na=False)]
        if not matching_rows.empty:
            return bigram, matching_rows.iloc[0]

    return None, None

def main():
    st.set_page_config(
        page_title="AI-based Laptop Recommendation and Negotiation Assistant",
        page_icon="💻",
        layout="wide"
    )
    
    # Add custom styling
    st.markdown(
        """
        <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #0073e6;
            text-align: center;
        }
        .section-header {
            font-size: 24px;
            font-weight: bold;
            color: #444;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .box {
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            background-color: #fff;
            color: #333;
        }
        .red-box {
            background-color: #545AA7;
            color: black;
        }
        .green-box {
            background-color: #4caf50;
            color: white;
        }
        .dark-blue-box {
            background-color: #1e3d58;
            color: white;
        }
        .sky-blue-box {
            background-color: #4C516D;
            color: black;
        }
        .yellow-box {
            background-color: #5F9EA0;
            color: white;
        }
        .icon-button {
            background-color: #0073e6;
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        }
        .icon-button:hover {
            background-color: #005bb5;
            cursor: pointer;
        }
        .icon {
            font-size: 20px;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown('<div class="title">Deal Recommendation and Negotiation Assistant</div>', unsafe_allow_html=True)

    # Initialize Google Sheets and load customer history
    credentials_path = "credentials2.json"
    spreadsheet_id = "1BQRX513_GAiLKokObbJoA30foTKzLC_BFS48LD5MMTM"
    customer_history_path = "C:\\Users\\choud\\Downloads\\dataset_preparation - Sheet1.csv"

    sheet = initialize_google_sheets(credentials_path, spreadsheet_id)
    customer_history = pd.read_csv(customer_history_path)

    # Input customer name
    customer_name = st.text_input("Enter Customer Name:", key="customer_name")
    
    # Display previous customer interaction if available
    if customer_name:
        previous_summary, deal_status = get_previous_interaction(customer_name, customer_history)
        if previous_summary:
            st.markdown(f"<div class='box sky-blue-box'><strong>Previous Summary:</strong> {previous_summary}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='box sky-blue-box'><strong>Deal Status:</strong> {deal_status}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='box sky-blue-box'>No previous conversation found for this customer.</div>", unsafe_allow_html=True)

    # Section for user input
    st.markdown('<div class="section-header">Customer Interaction</div>', unsafe_allow_html=True)
    
    # Create two columns for audio recording and text input
    col1, col2 = st.columns([1, 2])

    with col1:
        # Add a button with an icon
        record_button = st.markdown(
            """
            <button class="icon-button">
                <span class="icon">🎙️</span>
            </button>
            """, unsafe_allow_html=True
        )
        record_clicked = st.button("Start")

    with col2:
        text_input = st.text_area("Text input here:", key="text_input", placeholder="Enter your requirements...")

    # Add a "Stop Listening" button
    stop_listening = st.button("Stop")

    # Initialize conversation list
    conversation = []
    focused_laptops = []

    # Process user input (audio or text)
    if record_clicked or text_input:
        if record_clicked:
            st.write("Recording...")
            audio_file = record_audio_chunk(duration=6)
            text = transcribe_audio(audio_file)
        else:
            text = text_input  # Use text from text input area

        if text:
            conversation.append(text)
            st.write(f"Customer Input: {text}")

            # Check if "Stop Listening" is clicked
            if stop_listening:
                st.write("Stopping the conversation...")
                summary = summarize_conversation(conversation)
                sentiment = analyze_sentiment(conversation[-1])
                final_deal_status = "closed" if sentiment == "Positive" else "not closed"

                st.markdown(f"<div class='box sky-blue-box'><strong>Summary of Interaction:</strong> {summary}</div>", unsafe_allow_html=True)
                st.write(f"Final Deal Status: {final_deal_status}")

                row_index = find_customer_row(sheet, customer_name)
                if row_index:
                    append_to_existing_customer_row(sheet, row_index, conversation, summary, sentiment, final_deal_status)
                else:
                    append_new_customer_row(sheet, customer_name, conversation, summary, sentiment, final_deal_status)

                st.write("Customer interaction logged successfully.")
                return

            # Analyze sentiment
            sentiment = analyze_sentiment(text)
            sentiment_class = "green-box" if sentiment == "Positive" else "red-box" if sentiment == "Negative" else "dark-blue-box"
            st.markdown(f"<div class='box {sentiment_class}'>Sentiment: {sentiment}</div>", unsafe_allow_html=True)

            # Match laptop name
            laptop_name, laptop_row = extract_laptop_name(text, laptop_data)
            if laptop_name:
                st.write(f"Matched Laptop: {laptop_name}")
                if "discount" in text.lower() or "over my budget" in text.lower():
                    if sentiment in ["Neutral", "Negative"]:
                        discounted_price = laptop_row["Discounted Price"]
                        st.write(f"Discounted Price for {laptop_name}: ₹{discounted_price:.2f}")
                else:
                    st.write(f"No discount applicable as sentiment is {sentiment}.")

            # Get deal recommendations
            answer, recommendations, focused_laptops = get_deal_recommendations(text, customer_name, laptop_data, focused_laptops)
            st.markdown(f"<div class='box dark-blue-box'>{answer}</div>", unsafe_allow_html=True)

            if recommendations:
                st.write("**Laptop Recommendations**:")
                for rec in recommendations:
                    st.markdown(f"<div class='box red-box'>- {rec['Product Name']}, {rec['RAM']}, {rec['SSD']}, {rec['Battery Life']}, {rec['OS']}, {rec['Final Price']}</div>", unsafe_allow_html=True)

            # Provide negotiation tips
            if sentiment in ["Neutral", "Negative"]:
                negotiation_tips = negotiation_coach(conversation)
                st.markdown(f"<div class='box yellow-box'>Negotiation Tips: {negotiation_tips}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

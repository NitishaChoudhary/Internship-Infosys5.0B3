import time
import pyaudio
import wave
import speech_recognition as sr
from transformers import pipeline
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import pandas as pd
import re
from groq import Groq
import cohere

# Initialize Models and Configurations
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
laptop_data = pd.read_csv("C:\\Users\\choud\\Downloads\\laptop_dataset_updated.csv")  # Path to the dataset
laptop_data = laptop_data.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
co = cohere.Client("oykOriKwgZYtOf2tj5LfgH4thANYaerQnuIlJjwm")
# Groq API Initialization
groq_client = Groq(api_key="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")  # Google Sheets Initialization
def initialize_google_sheets(credentials_path, spreadsheet_id):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(spreadsheet_id).sheet1
    return sheet
# Speech-to-Text
def record_audio_chunk(duration=6, file_name="buyer_audio.wav"):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100
    
    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    print("Recording...")
    frames = [stream.read(chunk) for _ in range(0, int(rate / chunk * duration))]
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    return file_name

def transcribe_audio(file_name):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_name) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return "Error during transcription"
# Sentiment Analysis
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    label = result[0]['label']
    return "Positive" if label == "LABEL_2" else "Neutral" if label == "LABEL_1" else "Negative"
# Summarize Conversation
def summarize_conversation(conversation_lines):
    messages = [{"role": "user", "content": '\n'.join(conversation_lines)}]
    system_message = {
        "role": "system",
        "content": "Summarize the entire customer interaction in a single concise sentence."
    }
    all_messages = [system_message] + messages

    conversation_stream = groq_client.chat.completions.create(
        messages=all_messages,
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False
    )
    return conversation_stream.choices[0].message.content.strip()
# Function to generate negotiation terms based on previous inputs
def generate_negotiation_terms_via_cohere(previous_input):
    # Adjust the prompt to only consider previous input and suggest negotiation terms
    prompt = f"""
    Given the previous conversation input: "{previous_input}",
    suggest effective negotiation terms or strategies that can be used to negotiate with a customer to a seller. The suggestions should focus on areas such as price, discount, alternatives, and exploring different options.
    """

    # Generate response using Cohere
    response = co.generate(
        model='command-light-nightly',
        prompt=prompt,
        max_tokens=100,
        temperature=0.7
    )
    
    return response.generations[0].text.strip()
# Main negotiation coach function that takes the previous input and generates suggestions
def negotiation_coach(previous_input):
    return generate_negotiation_terms_via_cohere(previous_input)
# Find and Append Customer Rows
def find_customer_row(sheet, customer_name):
    try:
        data = sheet.get_all_values()  # Get all rows
        for index, row in enumerate(data):
            if row and row[0].strip().lower() == customer_name.strip().lower():  # Case-insensitive match
                return index + 1  # Return the row index (1-based index)
    except Exception as e:
        print(f"Error finding customer row: {e}")
    return None

def append_to_existing_customer_row(sheet, row_index, numbered_inputs, chat_summary, sentiment_result, deal_status):
    try:
        existing_row = sheet.row_values(row_index)  # Get current row data
        column_headers = sheet.row_values(1)  # Get column headers
        num_existing_interactions = (len(existing_row) - 1) // 4  # Exclude the customer name from the count
        next_interaction_number = num_existing_interactions + 1

        # Determine starting column index for the next interaction
        next_column_start = (next_interaction_number - 1) * 4 + 2  # Offset for contexts, summary, etc.

        # Ensure there are enough column headers for the next interaction
        required_headers = [
            f"{next_interaction_number}_Contexts",
            f"{next_interaction_number}_Summary",
            f"{next_interaction_number}_Sentiment",
            f"{next_interaction_number}_Deal_status",
        ]
        if len(column_headers) < next_column_start + 3:
            for i, header in enumerate(required_headers):
                sheet.update_cell(1, next_column_start + i, header)

        # Update the corresponding columns for the new interaction
        new_data = [
            '\n'.join(numbered_inputs),  # Contexts
            chat_summary,  # Summary
            sentiment_result,  # Sentiment
            deal_status,  # Deal status
        ]
        for i, data in enumerate(new_data):
            sheet.update_cell(row_index, next_column_start + i, data)
    except Exception as e:
        print(f"Error appending data to existing customer row: {e}")

def append_new_customer_row(sheet, customer_name, numbered_inputs, chat_summary, sentiment_result, deal_status):
    try:
        new_row = [
            customer_name,  # Customer name
            '\n'.join(numbered_inputs),  # Customer inputs
            chat_summary,  # Chat summary
            sentiment_result,  # Sentiment analysis
            deal_status,  # Deal status
        ]
        sheet.append_row(new_row)
    except Exception as e:
        print(f"Error appending new customer row: {e}")
# Google Sheets Updates
def update_google_sheets(sheet, customer_name, conversation, summary, sentiment, deal_status):
    row = [customer_name, conversation, summary, sentiment, deal_status]
    sheet.append_row(row)

# Load Customer Interaction History
def load_customer_history(history_path):
    return pd.read_csv(history_path)

def get_previous_interaction(customer_name, customer_history):
    customer_data = customer_history[customer_history["Customer Name"].str.lower() == customer_name.lower()]
    if not customer_data.empty:
        previous_summary = customer_data.iloc[-1]["Summary"]
        deal_status = customer_data.iloc[-1]["Deal_status"]
        return previous_summary, deal_status
    return None, None
# Helper function to extract bigrams from text
def extract_bigrams_from_text(text):
    text = re.sub(r"[^\w\s]", "", text.lower())  # Remove non-alphanumeric characters
    words = text.split()
    return [' '.join([words[i], words[i + 1]]) for i in range(len(words) - 1)]
def find_matching_laptops(context_text, laptop_names):
    # Extract bigrams from context text
    bigrams = extract_bigrams_from_text(context_text)
    single_words = set(context_text.lower().split())  # Extract single words from context

    matched_laptops = set()
    for name in laptop_names:
        name_lower = name.lower()
        
        # Check bigram matches
        if any(bigram in name_lower for bigram in bigrams):
            matched_laptops.add(name)
            continue
        
        # Check single-word matches
        if any(word in name_lower for word in single_words):
            matched_laptops.add(name)

    return list(matched_laptops)
def get_deal_recommendations(user_input, user_name, laptop_data, focused_laptops=None):
    try:
        # Initialize focused laptops if not provided
        if focused_laptops is None:
            focused_laptops = []

        # Cohere model to generate an answer to the query
        cohere_response = co.generate(
            model='command-light-nightly',
            prompt=(f"You are an assistant. Answer the following query based on the laptop dataset or general knowledge.concentrate on the terms which are asked for and answer accordingly.\nQuery: {user_input}\nAnswer:"),
            max_tokens=100,
            temperature=0.7
        )
        answer = cohere_response.generations[0].text.strip()

        # Extract budget from user input
        budget = None
        if "under" in user_input.lower() or "below" in user_input.lower():
            budget_matches = re.findall(r"(under|below)\s+([\w\s]+)", user_input.lower())
            if budget_matches:
                budget = words_to_numbers(budget_matches[0][1])
                print(f"Extracted budget: {budget}")  # Debugging output

        # Match laptops from the input
        laptop_names = laptop_data["Product Name"].tolist()
        matched_laptops = set()

        # Single-word matching
        single_words = set(user_input.lower().split())
        single_word_matches = [name for name in laptop_names if any(word in name.lower() for word in single_words)]
        matched_laptops.update(single_word_matches)

        # Bigram matching (if single-word matches don't cover all)
        if len(matched_laptops) < 8:
            bigrams = extract_bigrams_from_text(user_input)
            bigram_matches = [name for name in laptop_names if any(bigram in name.lower() for bigram in bigrams)]
            matched_laptops.update(bigram_matches)

        # Combine matched laptops and focused laptops for recommendations
        all_matched_laptops = set(focused_laptops + list(matched_laptops))

        # Filter the dataset for matching laptops
        if all_matched_laptops:
            filtered_laptops = laptop_data[laptop_data["Product Name"].isin(all_matched_laptops)]
        else:
            filtered_laptops = laptop_data.copy()

        # Apply budget filter if available
        if budget is not None:
            filtered_laptops = filtered_laptops[filtered_laptops["Final Price"] <= budget]
            print(f"Filtered laptops under {budget}: {len(filtered_laptops)} found")  # Debugging output

        # Limit recommendations to up to 8 or 10 laptops
        recommendations = [] if filtered_laptops.empty else filtered_laptops.head(10).apply(
            lambda row: {
                "Product Name": row["Product Name"],
                "Final Price": f"₹{row['Final Price']:.2f}",
                "RAM": f"{row['RAM']}GB",
                "SSD": f"{row['SSD (GB)']}GB",
                "Battery Life": f"{row['Battery Life (hrs)']} hrs",
                "OS": row["OS"],
            }, axis=1
        ).tolist()

        # Update focused laptops if matched laptops exist
        if matched_laptops:
            focused_laptops.clear()
            focused_laptops.extend(matched_laptops)

        if recommendations:
            return answer, recommendations, focused_laptops
        else:
            return answer, [], focused_laptops

    except Exception as e:
        print(f"Error in get_deal_recommendations: {e}")
        return "", [], focused_laptops
def words_to_numbers(word):
    word = word.lower().replace(",", "").replace("rs", "").strip()
    multiplier = {"lakh": 100000, "thousand": 1000, "hundred": 100}
    number_map = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
        "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90  
    }
    
    total = 0
    temp_number = 0

    # Split the input into words and handle numbers
    words = word.split()

    for part in words:
        if part.isdigit():  # Handle digit cases (e.g., "50000")
            total += int(part)
        elif part in number_map:  # Handle word numbers (e.g., "fifty")
            temp_number += number_map[part]
        elif part in multiplier:  # Handle multipliers (e.g., "thousand")
            total += temp_number * multiplier[part]
            temp_number = 0  # Reset for next group of numbers
        else:
            pass  # Ignore unknown words, which may be currency symbols, etc.

    # Adding the remaining temporary number
    total += temp_number

    return total if total > 0 else None
def extract_budget(user_input):
    budget = None
    if "under" in user_input.lower() or "below" in user_input.lower():
        budget_matches = re.findall(r"(under|below)\s+([\w\s]+)", user_input.lower())
        if budget_matches:
            budget = words_to_numbers(budget_matches[0][1])
            print(f"Extracted budget: {budget}")  # Debugging output
    return budget
def main():
    credentials_path = "credentials2.json"
    spreadsheet_id = "1BQRX513_GAiLKokObbJoA30foTKzLC_BFS48LD5MMTM"
    customer_history_path = "C:\\Users\\choud\\Downloads\\dataset_preparation - Sheet1.csv"  # Path to customer history

    sheet = initialize_google_sheets(credentials_path, spreadsheet_id)
    customer_history = pd.read_csv(customer_history_path)  # Adjust for your customer history format

    customer_name = input("Enter customer name: ")

    # Fetch previous interaction from customer history
    previous_summary, deal_status = get_previous_interaction(customer_name, customer_history)
    if previous_summary:
        print(f"Previous Conversation Summary:\n{previous_summary}\n Deal status:{deal_status}")
    else:
        print("No previous conversation found for this customer.")
        previous_summary = ""  # Initialize if no previous summary exists
    
    conversation = []
    focused_laptops = []  # Track customer focus on specific laptops
    
    while True:
        audio_file = record_audio_chunk()
        text = transcribe_audio(audio_file)
        if not text:
            continue
        conversation.append(text)
        print(f"Customer Input: {text}")
        sentiment = analyze_sentiment(text)
        intent = "purchase" if "buy" in text.lower() else "inquiry"

        answer, recommendations, focused_laptops = get_deal_recommendations(text, customer_name, laptop_data, focused_laptops)
        print(f"Answer: {answer}")
        print("Recommendations:")
        for rec in recommendations:
            print(f"- {rec['Product Name']}, {rec['RAM']}, {rec['SSD']}, {rec['Battery Life']}, {rec['OS']}, {rec['Final Price']}")

        if any(keyword in text.lower() for keyword in ["price", "discount", "alternatives", "exploring options"]):
            negotiation_tips = negotiation_coach(conversation)
            print(f"Negotiation Tips: {negotiation_tips}")

        if input("Type 'end' to finish: ").lower() == 'end':
            break

    summary = summarize_conversation(conversation)
    deal_status = "closed" if sentiment == "Positive" else "not closed"

    # Update the sheet
    row_index = find_customer_row(sheet, customer_name)
    if row_index:
        append_to_existing_customer_row(sheet, row_index, conversation, summary, sentiment, deal_status)
    else:
        append_new_customer_row(sheet, customer_name, conversation, summary, sentiment, deal_status)

if __name__ == "__main__":
    main()
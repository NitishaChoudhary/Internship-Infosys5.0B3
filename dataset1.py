import time
import pyaudio
import wave
import speech_recognition as sr
from transformers import pipeline
from groq import Groq
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Replace with your actual API key for Groq
client = Groq(api_key="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")
# Initialize the sentiment analyzer
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def record_audio_chunk(chunk_duration=6, file_name="buyer_audio.wav"):
    """Records a chunk of audio for the specified duration."""
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    rate = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("Listening for the buyer's input...")
    frames = []
    for _ in range(0, int(rate / chunk * chunk_duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_name, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    return file_name

def transcribe_audio(file_name="buyer_audio.wav"):
    """Transcribes audio to text using Google Speech Recognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_name) as source:
        print("Processing audio transcription...")
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print(f"Transcription: {text}")
            return text
        except sr.UnknownValueError:
            print("No clear speech detected in the last chunk.")
            return ""
        except sr.RequestError as e:
            print(f"Error in request: {e}")
            return ""

def analyze_sentiment(text):
    """Analyzes sentiment of the given text and returns Positive, Neutral, or Negative."""
    result = sentiment_analyzer(text)
    sentiment = result[0]["label"]
    if sentiment == "LABEL_2":
        return "Positive"
    elif sentiment == "LABEL_1":
        return "Neutral"
    else:
        return "Negative"


def summarize_conversation_single_line(conversation_lines):
    """Summarizes the entire conversation into a single line."""
    messages = [{"role": "user", "content": '\n'.join(conversation_lines)}]
    system_message = {
        "role": "system",
        "content": "Summarize the entire customer interaction in a single concise sentence."
    }
    all_messages = [system_message] + messages

    
    conversation_stream = client.chat.completions.create(
        messages=all_messages,
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False
    )
    return conversation_stream.choices[0].message.content.strip()

def initialize_google_sheets(credentials_path):
    """Initializes the Google Sheets integration."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(credentials)
    return client

def find_customer_row(sheet, customer_name):
    """
    Finds the row of the given customer name in the Google Sheet, case-insensitively.
    """
    try:
        data = sheet.get_all_values()  
        for index, row in enumerate(data):
            if row and row[0].strip().lower() == customer_name.strip().lower():  
                return index + 1  
    except Exception as e:
        print(f"Error finding customer row: {e}")
    return None

def append_to_existing_customer_row(sheet, row_index, numbered_inputs, chat_summary, sentiment_result, deal_status):
    """
    Appends new data under the appropriate numbered column for an existing customer row.
    """
    try:
        existing_row = sheet.row_values(row_index)  
        column_headers = sheet.row_values(1)  
        num_existing_interactions = (len(existing_row) - 1) // 4  
        next_interaction_number = num_existing_interactions + 1

       
        next_column_start = (next_interaction_number - 1) * 4 + 2  

        
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
    """Creates a new row for a new customer in the Google Sheet."""
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

def main():
    """Main workflow for continuous buyer interaction."""
    credentials_path = "credentials1.json"  # Path to your Google credentials file
    spreadsheet_id = "1UPPDEfSS8QMuFzPYPuVvTs2Ai3ffu1W49n-1ROqvPhA"  # Replace with your spreadsheet ID
    sheets_client = initialize_google_sheets(credentials_path)
    sheet = sheets_client.open_by_key(spreadsheet_id).sheet1

    print("Starting continuous buyer interaction...")

    buyer_name = input("Please enter the buyer's name: ")
    numbered_inputs = []
    customer_inquiries = []

    while True:
        audio_file = record_audio_chunk()
        transcribed_text = transcribe_audio(audio_file)
        if not transcribed_text:
            print("Waiting for the next buyer input...\n")
            time.sleep(4)
            continue

        numbered_inputs.append(f"{len(numbered_inputs) + 1}. {transcribed_text}")
        customer_inquiries.append(transcribed_text)

        print(f"\nBuyer Transcription: {transcribed_text}")
        print("Waiting for the next interaction...\n")
        end_chat = input("Type 'end' to finalize the chat or press Enter to continue: ")
        if end_chat.lower() == 'end':
            break

    last_line = customer_inquiries[-1] if customer_inquiries else ""

    sentiment_result = analyze_sentiment(last_line)
    chat_summary = summarize_conversation_single_line(customer_inquiries)
    deal_status = "closed" if sentiment_result == "Positive" else "not closed"

    # Check if customer already exists in the sheet
    customer_row_index = find_customer_row(sheet, buyer_name)
    if customer_row_index:
        append_to_existing_customer_row(sheet, customer_row_index, numbered_inputs, chat_summary, sentiment_result, deal_status)
    else:
        append_new_customer_row(sheet, buyer_name, numbered_inputs, chat_summary, sentiment_result, deal_status)

    print("Chat session finalized and logged in Google Sheets.")

if __name__ == "__main__":
    main()

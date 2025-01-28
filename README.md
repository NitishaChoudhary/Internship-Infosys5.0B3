Real-Time AI-Powered Sales Intelligence Tool
Introduction
The Real-Time AI-Powered Sales Intelligence Tool is designed to revolutionize live sales calls by providing actionable insights and suggestions to sales teams in real-time. This cutting-edge tool leverages advanced AI models for sentiment and intent analysis, integrates with CRM data and Google Sheets, and provides optimized negotiation strategies to enhance customer engagement and drive sales outcomes.

Features
Real-time Speech Recognition for continuous audio recording from the buyer.

Automated Speech-to-Text Transcription to convert recorded audio into text.

Sentiment and Intent Analysis powered by state-of-the-art NLP models.

Intelligent Negotiation Terms Generation based on buyer sentiment and intent.

Seamless Google Sheets Integration to record and manage buyer interactions.

Fully functional Workflow Integration to streamline processes end-to-end.

Project Workflow
The project workflow includes six major steps:

1. Speech Recognition
Tools Used: PyAudio, SpeechRecognition libraries.

Functionality: Continuously records audio input from the buyer during live sales calls.

2. Transcription of Recorded Audio
API: Google Speech-to-Text API.

Implementation:

Enabled Google Drive and Google Sheets APIs.

Downloaded credentials.json file using a service account on Google Console.

Output: Converts audio into text for further processing.

3. Sentiment and Intent Analysis
Models Used:

Sentiment Analysis: cardiffnlp/twitter-roberta-base-sentiment from Hugging Face.

Intent Analysis: facebook/bart-large-mnli from Hugging Face.

Objective: Understand the buyer's mood and intent to provide actionable insights.

4. Deal Recommendations
Approach: Suggest laptops based on the input given by the buyer and extract matched laptops names in the input to the product name in the dataset, recommends to the buyer.

5. Negotiating Terms Generation
Approach: Extracts keywords based on sentiment and intent analysis.

Generates basic negotiation terms tailored to the buyer's context.

6. Summarization of Conversation
Aprroach: Sumamrizes the whole conversation and finalises the deal status based on the sentiment.

Model used: llama 3.3 70b versatile model from GROQ LLM.

7. Google Sheets Integration
Purpose: Records all buyer interactions and contextual data for tracking and analysis.

Implementation:

Used the spreadsheet ID of a shared Google Sheet linked with the service account in the credentials file.

8. Workflow Integration
Objective: Seamlessly integrates all steps into a unified, functional workflow for real-time operation.

Contribution Guidelines
We welcome contributions to enhance this project. Please follow the guidelines below:

Reporting Issues
Check existing issues before creating a new one.

Provide a clear and concise description of the problem. Include steps to reproduce the issue, if applicable.

Install required dependencies:
pip install -r requirements.txt

Run the project:

python main.py

Acknowledgments
This project uses:

Hugging Face Transformers

Google Speech-to-Text API

Google Sheets API

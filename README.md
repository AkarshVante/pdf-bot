# pdf-bot
PDF Q&A Chatbot
A Streamlit-based application that allows users to upload a PDF, extract its text, and ask questions about its content using AI.
Features

Upload and process text-based PDFs.
Handles corrupt or image-only PDFs with clear error messages.
Uses AstraDB as a serverless vector database.
Supports OpenAI, Anthropic, and HuggingFace embeddings/LLMs with automatic fallbacks.
Polished UI with spinners, banners, and expandable sections.

Prerequisites

Python 3.10
AstraDB account with secure connect bundle
API keys for OpenAI and/or Anthropic (optional for HuggingFace)

Setup
Local Development

Install Python 3.10: Ensure Python 3.10 is installed on your system.
Clone the Repository: git clone <repo-url> && cd pdf_qa_bot
Create a Virtual Environment:
Linux/Mac: python3.10 -m venv venv && source venv/bin/activate
Windows: python3.10 -m venv venv && venv\Scripts\activate


Install Dependencies: pip install -r requirements.txt
Configure Environment Variables:
Copy .env.example to .env: cp .env.example .env
Edit .env with your AstraDB credentials and API keys.


Download Secure Connect Bundle:
Obtain it from your AstraDB dashboard.
Place it in the project root and update ASTRA_DB_SECURE_BUNDLE_PATH in .env.


Run the App: streamlit run app.py

Streamlit Cloud Deployment

Push to GitHub: Commit and push the project to a GitHub repository.
Connect to Streamlit Cloud:
Log in to Streamlit Cloud.
Link your GitHub repository.


Set Secrets:
In the Streamlit Cloud dashboard, go to your app's settings.
Add secrets matching the .env.example variables (e.g., ASTRA_DB_TOKEN, OPENAI_API_KEY).
Upload the secure connect bundle and note its path is not needed here; use a relative path like secure-connect-bundle.zip if required.


Deploy: Start the deployment process in Streamlit Cloud.

Usage

Open the app in your browser.
Upload a PDF file using the file uploader.
Once processed, enter a question in the text input and click "Submit".
View the AI-generated answer and expand "Most Relevant Passage" for context.

Troubleshooting

Rate Limits: If you hit rate limits (e.g., OpenAI), the app falls back to Anthropic or HuggingFace. Adjust EMBEDDING_PROVIDER or LLM_PROVIDER in .env if needed.
AstraDB Connectivity:
Missing/Invalid Token/ID: Verify ASTRA_DB_TOKEN and ASTRA_DB_ID in your .env or secrets.
Network Failure: Check your internet connection and AstraDB status. Ensure the secure connect bundle is valid.


PDF Errors: Ensure the PDF is text-based, not scanned images. Try a different file if corrupt.
Python Version: The app requires Python 3.10. Higher versions (e.g., 3.12) may cause compatibility issues.

Project Structure
pdf_qa_bot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python version specification
├── .env.example           # Template for environment variables
├── .gitignore             # Git ignore rules
├── .streamlit/
│   └── config.toml        # Streamlit configuration
└── README.md              # This file

# Computer Engineering AI Assistant

A Streamlit-based interface for a Lamini-trained LLM model designed to assist undergraduate computer engineering students.

## Features

- Interactive chat interface
- Specialized in computer engineering topics
- User-friendly design
- Real-time responses

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Lamini API key:
   ```
   LAMINI_API_KEY=your_api_key_here
   ```

## Running the Application

To run the application, use the following command:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` by default.

## Usage

1. Open the application in your web browser
2. Type your computer engineering-related question in the chat input
3. Wait for the AI assistant to respond
4. Continue the conversation as needed

## Note

Make sure you have a valid Lamini API key and your trained model is properly configured before using the application. 
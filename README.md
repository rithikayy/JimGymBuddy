# Meet Jim, Your Gym Buddy 💪 
Say hello to Jim, your personal fitness companion! Jim will help you achieve your fitness dreams, whether you need help setting your routine, or sticking with a consistent diet. 

## What Jim Can Help You With
- Routine recommendations
- Nutrition and diet assistance
- Exercise form and technique

## How Jim Works
Underneath the hood, Jim is an AI-powered assistant that answers workout questions using Retrieval-Augmented Generation (RAG) for context-specific responses. 
- Scraped 100+ documents from fitness wiki, processed into 400+ searchable chunks
- Uses OpenAI embeddings to find relevant information from the prompt
- Uses LangChain to incorporate retrieved information into GPT-4 responses
- Maintains conversation history for followups

## Tech Stack
- **LangChain**: For RAG and agent framework
- **OpenAI API**: Used for Embeddings and LLM
- **Streamlit**: For clean UI
- **BeautifulSoup**: Used for web scraping
- **Python**

## Setup
Want to try Jim for yourself? Follow these steps:
1. Clone the repository
   ```sh
   git clone https://github.com/rithikayy/JimGymBuddy.git
   ```
2. Get your Open AI API Key from: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
3. In your .env file, add your key
   ```sh
   OPENAI_API_KEY="your-key-here..."
   ```
5. Install requirements
   ```sh
   pip install -r requirements.txt
   ```
6. Run the command to talk to Jim!
   ```sh
   streamlit run app.py
   ```

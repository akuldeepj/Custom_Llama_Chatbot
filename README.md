
# Custom Chatbot with LangChain

This project demonstrates how to create a chatbot using LangChain. The chatbot is capable of loading and processing documents from a PDF file, creating a FAISS database for efficient retrieval, and interacting with users using an LLM (Large Language Model). The chatbot can handle greetings, goodbyes, and small talk, and answer questions based on the provided context from the PDF.

## Requirements

- Python 3.8 or higher
- `langchain_community`
- `langchain_core`
- `langchain_chains`
- `PyPDFLoader`
- `FAISS`
- `OllamaEmbeddings`
- `Ollama`

## Installation

1. Install Ollama:
   Visit [Ollama site](https://ollama.com/download) and download it according to your OS
   - [Mac](https://ollama.com/download/Ollama-darwin.zip)
   - [Windows](https://ollama.com/download/OllamaSetup.exe)
   - Linux:
      ```sh
      curl -fsSL https://ollama.com/install.sh | sh
      ```
   After the successful installation open up the terminal and run:
   ```sh
   ollama run llama3
   ```
   This will download the llama3-8B llm automatically

3. Clone the repository:

   ```sh
   git clone https://github.com/akuldeepj/Custom_Llama_Chatbot/
   cd Custom_Llama_Chatbot
   ```

4. Create a virtual environment and activate it:

   ```sh
   python -m venv env
   source env/bin/activate
   ```

5. Install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Place the PDF file you want to use in the project directory. For this example, the file should be named `Chatbot.pdf`.

2. Run the chatbot script:

   ```sh
   python Script_chat.py
   ```

3. Interact with the chatbot through the command line. Type your questions and the chatbot will provide answers based on the content of the PDF. Type "bye" to exit the chat.


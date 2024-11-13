This project is a web application built with Streamlit that allows users to interact with PDF documents using advanced AI capabilities provided by Google's Gemini model. The application enables users to upload PDF files, process them, and ask questions about the content of the PDFs.

install requirements:-
pip install streamlit pypdf2 python-docx pandas langchain-google-genai langchain-community faiss-cpu python-dotenv

## How It Works

1. **Upload PDFs**: Use the sidebar to upload one or more PDF files.
2. **Process Files**: Click "Submit & Process" to extract and process the text from the PDFs.
3. **Ask Questions**: Enter a question in the main content area to get answers based on the PDF content.

 ## Troubleshooting

- **API Key Error**: Ensure your `.env` file contains a valid Google API key.
  
1. **Run the Streamlit app with below command i windos**:

   ```sh
   streamlit run code2.py

import datetime
import os
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from llm import ClaudeHaikuLLM
from embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA


def load_pdf_files(directory: str):
    """
    Load and extract text from PDF files in the given directory.
    """
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            try:
                reader = PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                documents.append(text)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return documents


def initialize_rag_system(documents_dir: str):
    # Step 1: Load documents
    docs = load_pdf_files(documents_dir)
    
    # Step 2: Preprocess documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents([Document(page_content=doc) for doc in docs])
    
    # Step 3: Initialize embeddings
    embeddings = OllamaEmbeddings(model="llama3.1")
    
    # Create a FAISS vector store from the documents and their embeddings
    vector_store = FAISS.from_documents(split_docs, embeddings)
    
    # Step 4: Initialize LLM
    llm = ClaudeHaikuLLM()
    
    # Step 5: Establish RAG pipeline
    prompt_template = """
    Instructions:
    You are an AI assistant specialized in providing mental health support. You have access to a large collection of documents related to mental health. You will be provided the result of a facial emotion recognition model. 
    Please analyze the detected emotion and provide a response to the user query using the retrieved documents. If the documents are not relevant, rely on your training data.
    
    Retrieved Documents:
    {context}
    
    Detected Emotion:
    {question}
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vector_store.as_retriever(), 
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain, vector_store


def rag_generate(query: str, qa_chain):
    answer = qa_chain.run(query)
    return answer


def retrieve_documents(query: str, vector_store):
    retrieved_docs = vector_store.as_retriever().invoke(query)
    return retrieved_docs


if __name__ == "__main__":
    print("Welcome to the RAG System!")
    print("Type 'exit' or 'quit' to terminate the program.\n")

    qa_chain, vector_store = initialize_rag_system("demo_docs")

    while True:
        query = input("Enter your query: ")
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        try:
            start_time = time.time()
            retrieved_docs = vector_store.as_retriever().invoke(query)

            answer = qa_chain.run(query)

            end_time = time.time()

            total_time = end_time - start_time

            print("\nGenerated Answer:\n")
            print(answer)
            print("\n" + "=" * 50 + "\n")

            print(f"total time: {total_time}")

            current_time = datetime.datetime.now().strftime('%m%d%H%M%S')
            with open(f'answer_{current_time}.txt', 'w', encoding='utf-8') as file:
                file.write(f"User Query: {query}\n")

                file.write("\nRetrieved Documents:\n")

                for idx, doc in enumerate(retrieved_docs, 1):
                    file.write(f"\nDocument {idx}:\n")
                    file.write(doc.page_content)
                    file.write("\n" + "-" * 40 + "\n")

                file.write("\nAnswer:\n")
                file.write(answer)
                file.write("\n" + "=" * 50 + "\n")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("\n" + "=" * 50 + "\n")

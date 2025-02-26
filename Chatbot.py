import os
from typing import List, Optional
from flask import Flask, render_template, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA

app = Flask(__name__)

# Khởi tạo chatbot PDF
class PDFChatbot:
    def __init__(self, pdf_directory: Optional[str] = None):
        # Định nghĩa các thành phần
        self.vector_store = None
        self.history_vector_store = None  # Thêm vector store cho lịch sử chat
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.qa_chain = None
        self.loaded_documents = []
        self.custom_system_prompt = "You are a helpful AI assistant specialized in analyzing PDF documents."
        self.pdf_directory = pdf_directory or './data'
        self.initialize_pdfs_from_directory()

    def load_pdf(self, file_paths: List[str]) -> List[Document]:
        """Tải nhiều tệp PDF và trả về các tài liệu."""
        all_documents = []
        try:
            for file_path in file_paths:
                loader = PyPDFLoader(file_path)
                all_documents.extend(loader.load())
            return all_documents
        except Exception as e:
            print(f"Error loading PDFs: {e}")
            return []

    def load_pdfs_from_directory(self) -> List[Document]:
        """Tải tất cả các tệp PDF từ thư mục đã chỉ định."""
        pdf_files = [os.path.join(self.pdf_directory, f) for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        return self.load_pdf(pdf_files)

    def split_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
        """Chia nhỏ tài liệu thành các phần nhỏ hơn."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_documents(documents)

    def create_vector_store(self, documents: List[Document]) -> Optional[Chroma]:
        """Tạo vector store từ tài liệu."""
        try:
            self.vector_store = Chroma.from_documents(documents, self.embeddings)
            return self.vector_store
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

    def create_qa_chain(self, system_prompt: Optional[str] = None) -> Optional[RetrievalQA]:
        """Tạo chuỗi hỏi đáp với prompt hệ thống tùy chọn."""
        if not self.vector_store:
            return None

        try:
            effective_system_prompt = system_prompt or self.custom_system_prompt
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro-latest", 
                temperature=0.1, 
                convert_system_message_to_human=True,
                system_prompt=effective_system_prompt,
                model_kwargs={
                    "max_output_tokens": 8192,  
                    "top_k": 10,
                    "top_p": 0.95
                }
            )

            retriever = self.vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 10}
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=retriever,
                return_source_documents=True
            )
            return self.qa_chain
        except Exception as e:
            print(f"Error creating QA chain: {e}")
            return None

    def store_chat_history(self, query: str, answer: str):
        """Lưu trữ lịch sử câu hỏi và câu trả lời vào vector store."""
        global conversation_history
        chat_history = f"User: {query}\nBot: {answer}"
        conversation_history += chat_history  # Cập nhật lịch sử toàn bộ cuộc trò chuyện
        
        try:
            # Sử dụng embed_documents để chuyển đổi văn bản thành vector
            chat_vector = self.embeddings.embed_documents([chat_history])[0]
            
            if self.history_vector_store is None:
                # Khởi tạo history vector store nếu chưa có
                self.history_vector_store = Chroma.from_embeddings([chat_vector], [chat_history])
            else:
                # Thêm dữ liệu vào history vector store đã có
                self.history_vector_store.add([chat_vector], [chat_history])
        except Exception as e:
            print(f"Error storing chat history: {str(e)}")

    def process_pdf_query(self, query: str) -> str:
        """Xử lý câu hỏi và trả về chỉ câu trả lời mà không có thông tin 'User:' và 'Bot:'."""
        if not self.qa_chain:
            return "Error: QA chain not initialized. Please load PDFs first."
        
        # Cập nhật ngữ cảnh trò chuyện
        context = conversation_history + f"\nUser: {query}"  # Thêm câu hỏi vào lịch sử
        try:
            response = self.qa_chain.invoke({"query": context})
            answer = response['result']
            return answer  # Trả về câu trả lời
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def initialize_pdfs(self, pdf_files: Optional[List[str]] = None, custom_system_prompt: Optional[str] = None):
        """Khởi tạo vector store và QA chain từ các PDF."""
        self.loaded_documents = []

        if pdf_files is None:
            documents = self.load_pdfs_from_directory()
        else:
            documents = self.load_pdf(pdf_files)

        if not documents:
            return "Error: Could not load PDFs"

        split_docs = self.split_documents(documents)
        self.loaded_documents = split_docs

        if not self.create_vector_store(split_docs):
            return "Error: Could not create vector store"

        if custom_system_prompt:
            self.custom_system_prompt = custom_system_prompt

        if not self.create_qa_chain(custom_system_prompt):
            return "Error: Could not create QA chain"

        return f"PDFs loaded and processed successfully!"

    def initialize_pdfs_from_directory(self):
        """Tự động tải các PDF khi ứng dụng bắt đầu."""
        self.initialize_pdfs()


# Lịch sử câu hỏi và trả lời
conversation_history = ""

def update_conversation(query_input):
    """Cập nhật câu hỏi và câu trả lời vào lịch sử trò chuyện mà không thêm thông tin dư thừa như 'User:' và 'Bot:'."""
    global conversation_history
    answer = chatbot.process_pdf_query(query_input)
    answer = answer.replace("\n", "<br>")
    conversation_history += f"<div class='chat-message user'><p>{query_input}</p></div>"
    conversation_history += f"<div class='chat-message bot'><p>{answer}</p></div>"
    
    return conversation_history

# Khởi tạo chatbot
chatbot = PDFChatbot(pdf_directory='./data')
chatbot.initialize_pdfs()

@app.route("/", methods=["GET", "POST"])
def index():
    global conversation_history
    if request.method == "POST":
        query_input = request.form["query"]
        conversation_history = update_conversation(query_input)
    return render_template("index.html", conversation_history=conversation_history)

if __name__ == "__main__":
    app.run(debug=True)

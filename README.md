# 🧠 PDF Summarizer & Query Tool

A powerful web application that allows you to **upload PDF documents**, **generate intelligent summaries**, and **ask questions** about the content using **advanced AI models**.

---

## ✨ Features

- 📄 **PDF Text Extraction**: Supports both text-based and scanned PDFs using OCR (Tesseract)
- 🧠 **Intelligent Summarization**: Structured summaries using state-of-the-art language models
- ❓ **Question Answering**: Ask questions and get contextual answers from the document
- 🔍 **Semantic Search**: Find relevant content using vector embeddings and FAISS
- 📝 **Multiple Summary Formats**: Structured, bullet-point, and executive summary options
- 💻 **Responsive Web Interface**: Clean UI optimized for both desktop and mobile
- ⏱️ **Real-time Processing**: Live progress tracking during PDF processing

---

## 🧰 Technology Stack

### 🔙 Backend

- **Flask** – Python web framework for building APIs
- **PyMuPDF (`fitz`)** – PDF text extraction
- **Tesseract** – Optical Character Recognition for scanned documents
- **Transformers** – Hugging Face library for AI model integration
- **FAISS** – Vector similarity search for semantic retrieval
- **LangChain** – Text splitting and document preprocessing

### 🔝 Frontend

- **HTML5 / CSS3** – Modern responsive UI
- **JavaScript** – Client-side logic and API interaction
- **Font Awesome** – Visual icons for enhanced UI

---

## 🤖 AI Models

- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2`  
- **Language Model**: `google/gemma-2b-it` for both summarization and question answering

---

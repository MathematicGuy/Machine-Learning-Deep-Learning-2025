import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2

def extract_text_from_pdf(pdf_path):
	with open(pdf_path, 'rb') as file:
		reader = PyPDF2.PdfReader(file)
		text = ""
		for page in reader.pages:
			text += page.extract_text()
	return text

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap
	)
	chunks = splitter.split_text(text)
	return chunks

def save_to_faiss(chunks, model_name='all-MiniLM-L6-v2', index_path='faiss_index.index'):
	model = SentenceTransformer(model_name)
	embeddings = model.encode(chunks)

	dimension = embeddings.shape[1]
	index = faiss.IndexFlatL2(dimension)
	index.add(np.array(embeddings))

	faiss.write_index(index, index_path)
	print(f"Index saved to {index_path}")

# Example usage
if __name__ == "__main__":
	pdf_path = "path/to/your/document.pdf"  # Replace with actual PDF path
	text = extract_text_from_pdf(pdf_path)
	chunks = chunk_text(text)
	save_to_faiss(chunks)
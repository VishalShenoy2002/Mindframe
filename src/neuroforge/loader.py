from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import os

def load_text(filename:str):
    """
    Loads content from a .txt file using TextLoader.
    Raises an error if the file is missing or has the wrong extension.
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")
    
    if not filename.endswith('.txt'):
        raise ValueError(f"Please provide file with .txt extenstion.")
    
    loader = TextLoader(file_path=filename)
    return loader.load()

def load_pdf(filename:str):
    """
    Loads content from a .pdf file using PDFPlumberLoader.
    Raises an error if the file is missing or has the wrong extension.
    """

    if os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")
    
    if not filename.endswith('.pdf'):
        raise ValueError(f"Please provide file with .pdf extension")
    
    loader = PDFPlumberLoader(file_path=filename)
    return loader.load()

def load_docx(filename:str):
    """
    Loads content from a .docx file using UnstructuredWordDocumentLoader.
    Raises an error if the file is missing or has the wrong extension.
    """
    
    if os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")
    
    if not filename.endswith('.pdf'):
        raise ValueError(f"Please provide file with .pdf extension")
    
    loader = UnstructuredWordDocumentLoader(file_path=filename)
    return loader.load()


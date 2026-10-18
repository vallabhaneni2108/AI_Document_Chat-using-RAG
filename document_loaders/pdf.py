# loading from text

# from langchain_community.document_loaders import TextLoader

# data = TextLoader("document_loaders/notes.txt")

# docs = data.load()

# one documemnt contain metadata and pagecontent in a list

# print(docs[0]) # with metadata
# print(docs[0].page_content) # without metadata
# print(len(docs))




# loading from pdf

# from langchain_community.document_loaders import PyPDFLoader

# data = PyPDFLoader("document_loaders/GRU.pdf")

# docs = data.load()

# print(len(docs))
# print(docs[14]) # to see 15th page document




# loading from Website

from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/macbook-pro/"

data = WebBaseLoader(url)
docs = data.load()
print(len(docs))
print(docs[0].page_content)


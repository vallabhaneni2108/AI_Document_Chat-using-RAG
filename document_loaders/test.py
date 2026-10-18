# split by characters

# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import CharacterTextSplitter

# splitter = CharacterTextSplitter(
#     separator="",
#     chunk_size = 10,
#     chunk_overlap = 1
# )


# data = TextLoader("document_loaders/notes.txt")

# docs = data.load()

# # one documemnt contain metadata and pagecontent in a list
# chunks = splitter.split_documents(docs) # we are splitting documents into chunks


# print(len(chunks)) 

# for i in chunks:
#     print(i.page_content)
#     print()
#     print()
#     print() # to create some empty spaces  




# split by tokens
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import TokenTextSplitter


# data = PyPDFLoader("document_loaders/GRU.pdf")

# docs = data.load()

# splitter = TokenTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 10,
# )

# chunks = splitter.split_documents(docs) # we are splitting documents into chunks

# print(len(chunks)) # we got 28 chunks with chunk size of 1000
# print(chunks[0].page_content)



# recursive splitting under split by characters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


data = PyPDFLoader("document_loaders/GRU.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10,
)

chunks = splitter.split_documents(docs) # we are splitting documents into chunks

print(len(chunks)) # we got 28 chunks with chunk size of 1000
print(chunks[0].page_content)
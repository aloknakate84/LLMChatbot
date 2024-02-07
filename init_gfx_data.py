import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Below code is to load a csv file.
"""
loader = DirectoryLoader('data/', glob="**/*.csv", show_progress=True, loader_cls=CSVLoader)
#loader = CSVLoader(file_path="./data/arc_gfx_Jan_2024.csv", mode="elements")

loader = CSVLoader(
    file_path="./data/arc_gfx_Jan_2024.csv",
    csv_args={
        "delimiter": ",",
        "fieldnames": ['Article Title', 'Content', 'External URL'],
    },
)
"""
loader = DirectoryLoader('data/Graphics', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
documents = loader.load()
# print the 1st metadata.
print(documents[0])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

vector_store = Chroma.from_documents(texts, embeddings, collection_metadata={"hnsw:space": "cosine"}, 
                                     persist_directory="stores/gfx_cosine")

print("Custom Vector Store Created.")
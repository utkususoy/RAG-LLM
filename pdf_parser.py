from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
import os

print(f"Dokumanlar yukleniyor...")
folder_path = './resources/'
def get_all_pdfs(folder_path):
    list_of_pages = []
    for file_ in os.listdir(folder_path):
        if file_.endswith('.pdf'):
            loader = PyPDFLoader(file_path=folder_path+file_)
            pages = loader.load_and_split()
            list_of_pages += pages
    return list_of_pages

docs = get_all_pdfs(folder_path=folder_path)
print(f"Toplam yuklenen dokuman: {len(docs)}")
 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=200,  
    length_function=len
)

all_splits = text_splitter.split_documents(docs)

# for doc in docs:
#     text_content = doc.page_content
#     text_chunks = splitter.split_text(text_content)
#     all_text_chunks.extend(text_chunks)

for chunk in all_splits:
    print(chunk)

model_path = "thenlper/gte-large"
embeddings = HuggingFaceInferenceAPIEmbeddings(model_name=model_path)

# https://www.youtube.com/watch?v=dUkiQ_WI92c&ab_channel=AIWithTarun
# https://github.com/lucifertrj/Awesome-RAG
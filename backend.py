import git
import os
import deeplake
import requests

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake


def processGitLink(git_link) -> None:
    last_name = git_link.split("/")[-1]
    clone_path = last_name.split(".")[0]
    print(clone_path)
    return clone_path


def clone_repo(git_link, clone_path):
    if not os.path.exists(clone_path):
        git.Repo.clone_from(git_link, clone_path)
        return


def extract_all_files(clone_path, docs, allowed_extensions):
    root_dir = clone_path
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            file_extension = os.path.splitext(file)[1]
            if file_extension in allowed_extensions:
                try:
                    loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                    docs.extend(loader.load_and_split())
                except Exception as e:
                    pass
    return


def chunk_files(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    num_texts = len(texts)
    return texts


def embed_deeplake(texts, deeplake_path, hf):
    db = DeepLake(dataset_path=deeplake_path, embedding_function=hf, overwrite=True)
    db.add_documents(texts)
    return db


def queryFun(API_URL, headers, prompt, retrieved_docs):
    payload = {
        "inputs": {
            "question": prompt,
            "context": f" This is the relevant code snippet: {retrieved_docs}",
        }
    }
    print(payload)
    response = requests.post(API_URL, headers=headers, json=payload)

    return response

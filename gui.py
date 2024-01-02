import streamlit as st
import backend
import deeplake

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import DeepLake


local = False
if local:
    from dotenv import load_dotenv

    load_dotenv()

clone_path = "cloned_repo/"
print(clone_path)

docs = []
allowed_extensions = [".py", ".ipynb", ".md", ".txt"]


# model_name = "mistralai/Mistral-7B-v0.1"
# API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
headers = {"Authorization": "Bearer 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"}

deeplake_path = f"hub://adismort/{backend.processGitLink('https://github.com/adismort14/dns-resolver-socket-programming.git')}"

# hf = HuggingFaceEmbeddings(model_name=model_name, cache_folder="C:\\Users\\ADITYA\\.cache\\torch\\sentence_transformers\\mistralai_Mistral-7B-v0.1")

hf = HuggingFaceEmbeddings()

st.title("SURDE-SURfe your coDE")

user_repo = st.text_input(
    "Enter Github Link to your public codebase",
)

if user_repo:
    st.write("You entered:", user_repo)

    repoName = backend.processGitLink(
        "https://github.com/adismort14/dns-resolver-socket-programming.git"
    )

    clone_path += repoName

    backend.clone_repo(
        "https://github.com/adismort14/dns-resolver-socket-programming.git", clone_path
    )

    st.write("Your repo has been cloned inside the working directory.")

    st.write(
        "Parsing the content and embedding it. This may take some time. Please wait!"
    )

    exists = deeplake.exists(deeplake_path)
    if exists:
        db = DeepLake(
            dataset_path=deeplake_path,
            read_only=True,
            embedding_function=hf,
        )
    else:
        backend.extract_all_files(clone_path, docs, allowed_extensions)
        texts = backend.chunk_files(docs)
        db = backend.embed_deeplake(texts, deeplake_path, hf)

    st.write("Done Loading. Ready to take your questions.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Type your question here."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown({query})
        retrieved_docs = db.similarity_search(query)
        response = backend.queryFun(API_URL, headers, query, retrieved_docs)
        print(response)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

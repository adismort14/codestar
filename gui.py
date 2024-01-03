import streamlit as st
import backend

# import deeplake
import kinda_final

# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import DeepLake

print("Restarted")

local = False
if local:
    from dotenv import load_dotenv

    load_dotenv()

clone_path = "cloned_repo/"

docs = []
allowed_extensions = [".py", ".ipynb", ".md", ".txt"]


# API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
# headers = {"Authorization": "Bearer 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"}

# hf = HuggingFaceEmbeddings()

st.set_page_config(
    page_title="codestar.",
    page_icon="🚀",
    layout="wide",
)

st.markdown(
    """
<style>
.big-font {
    font-size:16px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title(
    ":blue[CODESTAR] - :blue[CODE S]urfing :blue[T]ool for :blue[A]nswer :blue[R]etrieval."
)

user_repo = st.text_input(
    r"$\textsf{\large Enter Github Link to your public codebase:}$",
    placeholder="https://github.com/<USERNAME>/<REPO_NAME>.git",
)

if user_repo:
    st.markdown(
        f'<p class="big-font">You entered: {user_repo}</p>'.format(user_repo=user_repo),
        unsafe_allow_html=True,
    )

    repoName = backend.processGitLink(user_repo)

    clone_path += repoName

    backend.clone_repo(user_repo, clone_path)

    # deeplake_path = f"hub://adismort/{repoName}"

    st.markdown(
        f'<p class="big-font">Your repo has been cloned inside the working directory.</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<p class="big-font">Parsing the content and embedding it. This may take some time. Please wait!</p>',
        unsafe_allow_html=True,
    )

    # exists = deeplake.exists(deeplake_path)
    # if exists:
    #     db = DeepLake(
    #         dataset_path=deeplake_path,
    #         read_only=True,
    #         embedding_function=hf,
    #     )
    # else:
    #     backend.extract_all_files(clone_path, docs, allowed_extensions)
    #     texts = backend.chunk_files(docs)
    #     db = backend.embed_deeplake(texts, deeplake_path, hf)

    docs = kinda_final.extract_all_files(clone_path, allowed_extensions)
    texts = kinda_final.chunk_files(docs)
    vectordb = kinda_final.create_vectordb(texts)
    qa_chain = kinda_final.retriever_pipeline(vectordb)

    st.markdown(
        f'<p class="big-font">Done Loading. Ready to take your questions.</p>',
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Type your question here."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        # retrieved_docs = db.similarity_search(query, k=5)
        # response = backend.queryFun(
        #     API_URL, headers, query, retrieved_docs, st.session_state.messages
        # )
        # response = backend.queryFunTrial(db, query)
        # print(type(response))
        # print(response)
        # print(response.content)

        # query = "What are the strengths of candidate, Aditya?"
        llm_response = qa_chain(query)
        sources = ""
        for source in llm_response["source_documents"]:
            sources += source.metadata["source"] + "\n"
        response = kinda_final.process_llm_response(llm_response)

        with st.chat_message("assistant"):
            st.markdown(response)
            st.markdown("Sources:")
            st.markdown(sources)
        st.session_state.messages.append({"role": "assistant", "content": response})

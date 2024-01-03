import streamlit as st
import backend
import kinda_final

print("Restarted")

local = False
if local:
    from dotenv import load_dotenv

    load_dotenv()

clone_path = "cloned_repo/"

docs = []
allowed_extensions = [
    ".py",
    ".ipynb",
    ".md",
    ".txt",
    ".c",
    ".cpp",
    ".js",
    ".json",
    ".html",
    ".css",
]

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

    st.markdown(
        f'<p class="big-font">Your repo has been cloned inside the working directory.</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<p class="big-font">Parsing the content and embedding it. This may take some time. Please wait!</p>',
        unsafe_allow_html=True,
    )

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
        llm_response = qa_chain(
            query + "If you are not sure of the answer, simply state so."
        )

        response = kinda_final.process_llm_response(llm_response)

        with st.chat_message("assistant"):
            st.markdown(response)
            st.markdown("Sources:")
            for source in llm_response["source_documents"]:
                st.markdown(source.metadata["source"])
        st.session_state.messages.append({"role": "assistant", "content": response})

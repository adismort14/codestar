import git
import os
import requests

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.chains import RetrievalQA
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import pipeline
import torch


def processGitLink(git_link) -> None:
    last_name = git_link.split("/")[-1]
    clone_path = last_name.split(".")[0]
    return clone_path


def clone_repo(git_link, clone_path):
    if not os.path.exists(clone_path):
        git.Repo.clone_from(git_link, clone_path)
        return


def extract_all_files(clone_path, docs, allowed_extensions):
    root_dir = clone_path
    for dirpath, _, filenames in os.walk(root_dir):
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
    return texts


def embed_deeplake(texts, deeplake_path, hf):
    db = DeepLake(dataset_path=deeplake_path, embedding_function=hf, overwrite=True)
    db.add_documents(texts)
    return db


def queryFun(API_URL, headers, prompt, retrieved_docs, session_messages):
    retrieved_page_content = []
    i = 1
    for doc in retrieved_docs:
        retrieved_page_content.append(
            "Relevant Code Snippet " + str(i) + ": " + doc.page_content
        )
        i += 1

    payload = {
        "inputs": {
            "question": prompt,
            "context": f" This is the relevant code snippet: {retrieved_page_content} and this is the previously asked questions of the user: {session_messages}",
        }
    }
    print(payload)
    response = requests.post(API_URL, headers=headers, json=payload)

    return response


def queryFunTrial(db, query):
    model_name = "Intel/dynamic_tinybert"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding=True, truncation=True, max_length=512
    )

    # question_answerer = pipeline(
    #     "question-answering", model=model_name, tokenizer=tokenizer, return_tensors="pt"
    # )

    # llm = HuggingFacePipeline(
    #     pipeline=question_answerer, model_kwargs={"temperature": 0.7, "max_length": 512}
    # )

    retriever = db.as_retriever(search_kwargs={"k": 4})
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False
    # )

    outp = retriever.get_relevant_documents(query)
    print(outp[1].page_content)
    context = outp[1].page_content

    inputs = tokenizer.encode_plus(query, context, return_tensors="pt")
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )

    print(answer)

    # result = qa.run({"query": query})
    # return result["result"]
    return answer

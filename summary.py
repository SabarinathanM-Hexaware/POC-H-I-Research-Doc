from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.chroma import Chroma
import time
from langchain_community.vectorstores.chroma import Chroma
import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

azure_endpoint = os.environ["OPENAI_ENDPOINT"]
deployment = os.environ["OPENAI_EMBEDDING_DEPLOYMENT"]
model = os.environ["OPENAI_EMBEDDING_MODEL"]
persist_directory = os.environ["CHROMA_PERSIST_DIRECTORY"]
model = os.environ["OPENAI_MODEL"]
azure_deployment = os.environ["OPENAI_DEPLOYMENT"]
azure_endpoint = os.environ["OPENAI_ENDPOINT"]

llm = AzureChatOpenAI(
    model=model,
    temperature=0,
    azure_deployment=azure_deployment,
    azure_endpoint=azure_endpoint,
)
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=azure_endpoint,
    deployment=deployment,
    model=model,
)

vector_store = Chroma(
    persist_directory=persist_directory, embedding_function=embeddings
)

template = """
I want you to act as an assistant who will analyze several clinical and drug research papers,
and provide a detailed summary about the subject that is being asked for. The context will
provided from various sources like PubMed, Drug bank, Dailymed, and FDA. It will contain
abstract about a disease, its causes, symptoms, preventive measures. Along with that
drug trials will also be available with its efficacy on the disease.

With this context, You have to analyze the Symptoms for the disease, Key findings from the clinical 
and drug trials, What are the current drugs available for the disease and their efficacy,
and What is the treatment for the disease.

After analysing you have to give me a summary with 3 subsections
1. A Complete summary about the subject 
2. Main outcomes and measures 
3. Study design of background


If you don't know the answer, just say that you don't know.
You don't need to provide the citations for the articles.
Don't give any answer outside the information that is not provided in the context.

THE OVERALL ARTICLE SHOULD BE AROUND 2000 WORDS
Subject: {subject}
Context: {context}
"""


SUMMARY_CACHE_PATH = "output_cache/summary_{uuid}.md"


def cache_summary_output(content: str):
    uuid = int(time.time())
    with open(SUMMARY_CACHE_PATH.format(uuid=uuid), "w") as file:
        file.write(content)


def summarize(question: str) -> str:
    related_docs = vector_store.similarity_search(question, k=60)
    prompt = PromptTemplate(template=template, input_variables=["subject", "context"])
    chain = load_summarize_chain(
        llm,
        chain_type="stuff",
        verbose=False,
        prompt=prompt,
        document_variable_name="context",
    )
    output = chain.invoke({"input_documents": related_docs, "subject": question})
    summary = output["output_text"]
    return summary

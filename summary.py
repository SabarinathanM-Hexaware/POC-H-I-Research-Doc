import os
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.chroma import Chroma

azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
deployment = os.environ["OPENAI_EMBEDDING_DEPLOYMENT"]
model = os.environ["OPENAI_EMBEDDING_MODEL"]
persist_directory = os.environ["CHROMA_PERSIST_DIRECTORY"]
model = os.environ["OPENAI_MODEL"]
azure_deployment = os.environ["OPENAI_DEPLOYMENT"]
azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]

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
You are an assistant who would help in creating a research document on a disease
and it's associated drug.

Subject: {subject}

Prompt: {sub_prompt}

Context: {context}

Instructions:
1. If you don't know the answer, please skip that part.
2. You don't need to provide the citations for the articles.
3. Don't give any answer outside the information that is not provided in the context.
4. Please give the output in 500 words minimum.
5. Please don't mention anything about the context, If anything is not availabe in the
   context, please move forward.
6. Generate the output that would fit in a Word Document (Docx)
7. Use UPPERCASE to emphasize the key points, Capitalize the statements appropriately.
"""


def summarize(question: str, sub_prompt: str) -> str:
    related_docs = vector_store.similarity_search(question, k=60)
    prompt = PromptTemplate(template=template, input_variables=["subject", "context"])
    chain = load_summarize_chain(
        llm,
        chain_type="stuff",
        verbose=False,
        prompt=prompt,
        document_variable_name="context",
    )
    output = chain.invoke(
        {"input_documents": related_docs, "subject": question, "sub_prompt": sub_prompt}
    )
    summary = output["output_text"]
    return summary

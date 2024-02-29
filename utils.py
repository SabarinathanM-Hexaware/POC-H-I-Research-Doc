from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain
from langchain_core.documents import Document
from langchain.chains.base import Chain
from langchain_openai import AzureChatOpenAI
from typing import Sequence, List
import os

os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
UNWANTED_URL_TAGS = ["script", "style"]
TOKENIZER_MODEL_TYPE = "gpt-35-turbo"
MODEL_NAME = "base-gpt35-turbo-16"
CHUNK_SIZE = 13000
PROMPT = "You are an web scraping tool that is extracting information from the web without modifying the content "


def load_html(urls: list[dict]) -> List[Document]:
    loader = AsyncChromiumLoader([
        url["url"] for url in urls
    ])
    html = loader.load()
    return html

def transform_html(html: List[Document], urls: list[dict]) -> Sequence[Document]:
    docs_transformed = []

    bs_transformer = BeautifulSoupTransformer()
    html_transformer = Html2TextTransformer()

    for i in range(len(html)):
        # docs_transformed.extend(
        #     bs_transformer.transform_documents(
        #         [html[i]],
        #         tags_to_extract=urls[i]["tags_to_extract"],
        #         unwanted_tags=[*UNWANTED_URL_TAGS, *urls[i]["unwanted_tags"]],
        #         remove_lines=True,
        #     )
        # )
        docs_transformed.extend(
            html_transformer.transform_documents(
                [html[i]]
        ))
    
    return docs_transformed

def split_documents_into_chunks(docs_transformed: Sequence[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=0,
    model_name=TOKENIZER_MODEL_TYPE
    )

    splits = splitter.split_documents(docs_transformed)
    return splits

def query(content: str, schema: dict, verbose = False) -> Chain:
    llm = AzureChatOpenAI(temperature=0, model=MODEL_NAME)
    return create_extraction_chain(schema=schema, llm=llm, verbose=verbose).run(content)

def merge_documents(docs: List[Document]) -> Document:
    
    return Document(
        page_content="\nAnother Source:\n".join([doc.page_content for doc in docs])
    )



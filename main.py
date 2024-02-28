import os
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ[
    "AZURE_OPENAI_ENDPOINT"
] = "https://oai-rapidx-finetune-dev.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "b261cb96ff9640fd8bfffdfc022e2c35"
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["OPENAI_EMBEDDING_DEPLOYMENT"] = "base-text-embedding-ada-002"
os.environ["OPENAI_DEPLOYMENT"] = "base-gpt35-turbo-16"
os.environ["OPENAI_MODEL"] = "gpt-35-turbo-16K"

from utils import (
    load_html,
    transform_html,
    split_documents_into_chunks,
    query,
    merge_documents,
)
from doc import create_summary_document
from langchain.chains.base import Chain
from typing import List
import json
from summary import summarize

urls = [
    {
        "url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=0ab267f4-9ec9-44fd-8521-f975074667d9&audience=consumer",
        "tags_to_extract": ["p"],
        "unwanted_tags": ["header"],
    },
    {
        "url": "https://go.drugbank.com/drugs/DB00067",
        "tags_to_extract": ["p", "dl", "dt", "dd", "tr", "div"],
        "unwanted_tags": [],
    }
]

section5_schema = {
    "properties": {
        "summary_of_related_research_articles": {"type": "string"},
        "main_outcomes_and_measures_of_related_articles": {"type": "string"},
        "study_design_for_background": {"type": "string"},
    },
    "required": [
        "summary_of_related_research_articles",
        "main_outcomes_and_measures_of_related_articles",
        "study_design_for_background",
    ],
}


schema = {
    "properties": {
        "summary": {"type": "string"},
        "background": {"type": "string"},
        "molecular_structure": {"type": "string"},
        "molecular_weight": {"type": "string"},
        "molecular_formula_of_drug": {"type": "string"},
        "indication_and_usage": {"type": "string"},
        "associated_conditions": {"type": "string"},
        "associated_therapies": {"type": "string"},
        "mechanism_of_action": {"type": "string"},
        "metabolism": {"type": "string"},
        "route_of_elimination": {"type": "string"},
        "half_life": {"type": "string"},
        "toxicity": {"type": "string"},
        "dose_and_administration": {"type": "string"},
        "contraindiacations_of_drug": {"type": "string"},
        "warnings_and_precautions": {"type": "string"},
        "clinical_pharmacology_of_drug": {"type": "string"},
        "non_clinicol_toxicology": {"type": "string"},
        "clinical_studies": {"type": "string"},
        "storage_and_handling": {"type": "string"},
    },
    "required": [
        "summary",
        "background",
        "molecular_structure",
        "molecular_weight",
        "molecular_formula_of_drug",
        "indication_and_usage",
        "associated_conditions",
        "associated_therapies",
        "mechanism_of_action",
        "metabolism",
        "route_of_elimination",
        "half_life",
        "toxicity",
        "dose_and_administration",
        "contraindiacations_of_drug",
        "warnings_and_precautions",
        "clinical_pharmacology_of_drug",
        "non_clinicol_toxicology",
        "clinical_studies",
        "storage_and_handling",
    ],
}


def write_to_file(content, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(content)


documents = load_html(urls)
transformed_docs = transform_html(documents, urls)

responses: List[Chain] = []
for doc in transformed_docs:
    print(f"Processing {doc.metadata}")
    splitted_chunks = split_documents_into_chunks([doc])
    print("*"*50)
    print(f"Splitting into {len(splitted_chunks)} chunks")
    resp = query(splitted_chunks[0].page_content, schema)
    responses.append(resp)

summarized_content = summarize("Summary of Acute Renal Failure and the related drugs")

res = query(summarized_content, section5_schema)

responses.append(res)

summary = dict()

for resp in responses:
    for key, value in resp.items():
        if (key not in summary) or (len(summary[key]) < len(value)):
            summary[key] = str(value).replace("\n", "\n\n")


write_to_file(json.dumps(summary, indent=4), "summary_now.json")
print("Creating Summary Document")
# create_summary_document(summary)

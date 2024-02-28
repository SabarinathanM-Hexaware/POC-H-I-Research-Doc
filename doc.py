import json
import time
import shutil
from pathlib import Path
from docx import Document
from typing import List

para_mapping = {
    "summary": 76,
    "background_of_drug": 78,
    "molecular_structure": 80,
    "molecular_weight": 82,
    "molecular_formula_of_drug": 84,
    "indication_and_usage": 86,
    "associated_conditions": 88,
    "associated_therapies": 90,
    "mechanism_of_action": 92,
    "metabolism": 94,
    "route_of_elimination": 96,
    "half_life": 98,
    "toxicity": 100,
    "dose_and_administration": 102,
    "contraindiacations_of_drug": 104,
    "warnings_and_precautions": 106,
    "clinical_pharmacology": 108,
    "non_clinicol_toxicology": 110,
    "clinical_studies_of_drug": 112,
    "storage_and_handling_of_drug": 114,
    "summary_of_related_research_articles": 157,
    "main_outcomes_and_measures_of_related_articles": 159,
    "study_design_for_background": 163,
}


def create_summary_document(summary: dict[str, str]) -> None:
    # Creating a new Summary document from template
    template_file = Path("Research_Doc_Template.docx")
    current_epoch_time = int(time.time())
    target_file_path = Path(f"output/Research_Doc_{current_epoch_time}.docx")
    shutil.copyfile(template_file, target_file_path)

    # Adding the summary text to the summary document
    doc = Document(target_file_path)
    paras = doc.paragraphs
    for [key, index] in para_mapping.items():
        if key in summary:
            paras[index].runs[0].text = summary[key].capitalize()
        else:
            paras[index].runs[0].text = ""

    doc.save(target_file_path)

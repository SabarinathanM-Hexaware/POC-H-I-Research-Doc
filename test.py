from summary import summarize
from utils import query

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
summarized_content = summarize(
    "Summary of Background Research for Vasopressin in Renal Failure"
)

res = query(summarized_content, section5_schema)
print(summarized_content, res)

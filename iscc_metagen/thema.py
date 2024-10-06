"""
# Thema Subject Categories Classification

## Category data structure
[
    {
        "CodeValue": "A",
        "CodeDescription": "The Arts",
        "CodeNotes": "Use all A* codes for: specialist and general adult titles, including ..."
        "CodeParent": "",
        "IssueNumber": 1,
        "Modified": 1.4
    },
    {
        "CodeValue": "AB",
        "CodeDescription": "The arts: general topics",
        "CodeNotes": "",
        "CodeParent": "A",
        "IssueNumber": 1,
        "Modified": 1.5
    },
    ...
]

## Strategies

### Top-Down LLM based Iterative Selection

Steps:
  - Given a text extract from the book:
    - Select 1 main and up to 3 secondary categories from top level categories
    - Do the same for SubCategories of the selected categories traversing down the tree
    - Select the main and up to 3 secondary headings from the list of collected leafs

### RAG Based Selection

Steps:
  - Given a text extract from the book:
    - Generate multiple search queries
    - Query categories from RAG index using questions
    - From each branch select the deepest category
    - Ask LLM the rerank and select top 4 categories
"""
import io

from loguru import logger as log
from functools import cached_property
from typing import Annotated, List, Literal
import httpx_cache
from pydantic import BaseModel, BeforeValidator, Field
from iscc_metagen.utils import timer
from iscc_metagen.prompt import make_prompt
import pathlib

HERE = pathlib.Path(__file__).parent.absolute()

THEMA_VERSION = "1.5"
THEMA_JSON_URL = "https://www.editeur.org/files/Thema/1.5/v1.5_en/20230707_Thema_v1.5_en.json"
THEMA_PATH = HERE / "thema.json"
NAMESPACE = "6ba7b811-9dad-11d1-80b4-00c04fd430c8"


# Thema JSON has a mix of number and string values for some of the fields
Str = Annotated[str, BeforeValidator(str)]


import json


class ThemaCode(BaseModel):
    category_code: Str = Field(
        alias="CodeValue", description="The unique code value for the Thema category"
    )
    category_heading: str = Field(
        alias="CodeDescription", description="The descriptive heading for the Thema category"
    )
    code_notes: str = Field(
        alias="CodeNotes",
        default="",
        description="Additional notes or information about the Thema category",
    )
    parent_code: Str = Field(
        alias="CodeParent",
        default="",
        description="The parent code of this category in the Thema hierarchy",
    )
    issue_number: Str = Field(
        alias="IssueNumber",
        default="",
        description="The issue number of the Thema version this code belongs to",
    )
    modified: Str = Field(
        alias="Modified", default="", description="The modification version of this Thema code"
    )

    class Config:
        populate_by_name = True

    def render_simple(self) -> str:
        """
        Returns a JSON string with indent 2 that includes the category_code and category_heading fields.
        """
        simple_dict = {
            "category_code": self.category_code,
            "category_heading": self.category_heading,
        }
        return json.dumps(simple_dict, indent=2)


class ThemaSelection(BaseModel):
    """A Thema category relevant to a specific docuemnt."""

    reason: str = Field(description="Reason for selection")
    category_code: str = Field(description="Category code")
    category_heading: str = Field(description="Category heading")
    confidence: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="How confident you are that the selected category is relevent"
    )


class ThemaCategories(BaseModel):
    """A list of Thema categories relevant to the docunent"""

    categories: List[ThemaSelection]


from iscc_metagen.client import client
from iscc_metagen.pdf import pdf_extract_pages
from iscc_metagen.settings import mg_opts


def predict_categories(doc):
    # type: (str|Path|Document) -> ThemaCategories
    """
    Predict Thema Categories for a document.

    :param doc: The document to analyze (file path or Document object)
    :return: ThemaCategories object containing the predicted categories
    """
    # Extract pages from the document
    pages = pdf_extract_pages(doc, first=8, middle=3, last=3)

    # Initialize Thema
    thema = Thema()

    # Get main subject categories as a formatted string
    categories = "\n".join(
        [f"{code.category_code}: {code.category_heading}" for code in thema.main_subjects]
    )

    # Generate the prompt
    prompt = prompt_select_category(pages=pages, categories=categories)

    # Use the AI model to predict categories
    response = client.chat.completions.create(
        model=mg_opts.litellm_model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that selects Thema categories for books.",
            },
            {"role": "user", "content": prompt},
        ],
        response_model=ThemaCategories,
    )

    return response


@make_prompt
def prompt_select_category(pages, categories):
    """
    You are tasked with selecting the most relevant Thema categories for a document based on
    excerpts from its beginning, middle, and end. Your goal is to choose 1 to 4 categories that
    best represent the document's content, ensuring the first category is the most relevant.

    Here is the list of Thema categories to choose from:
    {{ categories }}

    Now, carefully read the following excerpts from the document:
    {{ pages }}

    Analyze these excerpts to understand the main themes, topics, and focus of the document.
    Consider the following:
    1. What are the primary subjects discussed?
    2. Are there any recurring themes or ideas?
    3. What is the overall tone or approach of the document?

    Based on your analysis, select 1 to 4 relevant Thema categories from the provided list.
    Remember:
    - Choose categories that best represent the document's content.
    - Ensure the first category you list is the most relevant and important.
    - Only select categories if they are truly applicable to the document.
    - It's acceptable to choose fewer than 4 categories if that better represents the document.

    Remember to base your selection and explanation solely on the provided excerpts and Thema
    categories.
    """


class Thema:
    def __init__(self):
        self.data = load_thema_json()
        self.codes = parse_thema_codes(self.data)
        self.db = {code.category_code: code for code in self.codes}

    @cached_property
    def main_subjects(self) -> list[ThemaCode]:
        """Returns a list of main subject headings"""
        return [
            code
            for code in self.codes
            if len(code.category_code) == 1 and code.category_code.isalpha()
        ]

    def sub_categories(self, parent: ThemaCode) -> list[ThemaCode]:
        """Returns a list of sub categories headings for a given category"""
        parent_code = parent.category_code
        return [
            code
            for code in self.codes
            if code.parent_code == parent_code and code.category_code != parent_code
        ]


def ingest_thema():
    """Ingest Thema Categories into Vector DB"""

    data = load_thema_json()
    codes = parse_thema_codes(data)
    docs = build_thema_docs(codes)
    metadata = [code.model_dump() for code in codes]
    # ids = [uuid.uuid5(NAMESPACE, code.CodeValue) for code in codes]
    with timer("Ingested thema categories:"):
        ids = vdb.add(
            collection_name="thema",
            documents=docs,
            metadata=metadata,
            batch_size=64,
        )
    return ids


def load_thema_json():
    # type: () -> dict
    """Load original Thema JSON"""
    try:
        with timer("Loading thema json from disk"):
            with open(THEMA_PATH, "rt", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"Error loading thema json from disk: {e}")
        with timer("Loading Thema Json"):
            with httpx_cache.Client(cache=httpx_cache.FileCache(), always_cache=True) as client:
                return client.get(THEMA_JSON_URL).json()


def save_thema_json():
    data = load_thema_json()
    with open(THEMA_PATH, 'w', newline='\n', encoding="utf-8") as file:
        json.dump(data, file, separators=(',', ':'), ensure_ascii=False)



def parse_thema_codes(data):
    # type: (dict) -> list[ThemaCode]
    """Parse Thema codes from raw JSON data"""
    with timer("Parsing Thema Codes"):
        codes = data["CodeList"]["ThemaCodes"]["Code"]
        return [ThemaCode.model_validate(code) for code in codes]


def build_thema_docs(codes):
    # type: (list[ThemaCode]) -> list[str]
    """Return list of documents composed of full headings and notes"""
    code_dict = {code.category_code: code for code in codes}
    docs = []

    def get_full_heading(code):
        # type: (ThemaCode) -> str
        """Recursively build the full heading for a code"""
        if code.CodeParent and code.CodeParent in code_dict:
            parent_heading = get_full_heading(code_dict[code.CodeParent])
            return f"{parent_heading} / {code.CodeDescription}"
        return code.CodeDescription

    for code in codes:
        doc = get_full_heading(code)
        if code.CodeNotes:
            doc += f"\n{code.CodeNotes}"
        docs.append(doc)
    return docs


if __name__ == "__main__":
    from rich import print as p
    pdf = HERE.parent / ".data/test1.pdf"
    p(predict_categories(pdf))

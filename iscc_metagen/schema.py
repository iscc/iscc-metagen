# -*- coding: utf-8 -*-
from typing import Literal, Optional, Annotated
from pydantic import BaseModel, Field, HttpUrl, conint, AfterValidator, field_validator
from pydantic.json_schema import SkipJsonSchema
from pydantic_extra_types.language_code import LanguageAlpha2
from pydantic_extra_types.isbn import ISBN


class Contributor(BaseModel):
    name: str = Field(..., description="The full name of the contributor")
    role: str = Field(..., description="The role of the contributor")


class BookISBN(BaseModel):
    isbn: ISBN = Field(
        ..., description="The ISBN number (Only the number without prefix or dashes)"
    )
    edition: Optional[str] = Field(None, description="The book edition to which the ISBN belongs")


class BookMetadata(BaseModel):
    title: str = Field(..., description="The title of the book")
    subtitle: Optional[str] = Field(None, description="The subtitle of the book")
    description: str = Field(..., description="A short and concise description of the book")
    keywords: list[str] = Field(
        ..., description="Keywords that apply to the books topic", min_length=3, max_length=7
    )
    publisher: Optional[str] = Field(..., description="The name of publisher of the book")
    publisher_website: Optional[HttpUrl] = Field(
        ..., description="Website URL of the publisher (including http/https prefix)"
    )
    year_published: Optional[conint(ge=1, le=9999)] = Field(
        ..., description="The year of publication"
    )
    language: LanguageAlpha2 = Field(
        ..., description="The language of the book (as ISO 639-1 alpha-2)"
    )
    contributors: Optional[list[Contributor]]
    isbns: Optional[list[BookISBN]]

    model: SkipJsonSchema[str] = Field("", description="Model used for generation")
    response_cost: SkipJsonSchema[float] = Field(0.0, description="Response cost in USD")

    # @field_validator("title")
    # def title_is_uppercase(cls, v: str):
    #     assert v.isupper(), "Title must be uppercase"
    #     return v


class PageType(BaseModel):
    """
    Classify the page-type of a book page. Also take note of the page-number when predicting the page type.
    For example, it is quite unlikely that page 0 is an IMPRINT.
    """

    chain_of_thought: str = Field(
        ..., description="The chain of thought that led to the prediction."
    )
    page_type: Literal[
        "TITLE_PAGE",
        "IMPRINT",
        "TABLE_OF_CONTENTS",
        "OTHER",
    ] = Field(..., description="The predicted page type.")
    confidence: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="The confidence score of your prediction."
    )
    page_number: int = Field(..., description="The page number")


class Page(BaseModel):
    page_type: PageType
    content: str = Field(..., description="Markdown representation of the pages textual content")

from pathlib import Path
from typing import List
import fitz
import pymupdf4llm
from iscc_metagen.schema import PageType, Page
from iscc_metagen.client import client
from iscc_metagen.settings import mg_opts
from loguru import logger as log


def get_page_type(text, pageno=None):
    # type: (str, int|None) -> PageType
    """Perform  single-label page-type classification"""
    if pageno is None:
        prompt = f"Classify the following page:\n\n<page_text>{text}</page_text>"
    else:
        prompt = (
            "Classify the following"
            f" page:\n\n<page_number>{pageno}</page_number>\n<page_text>{text}</page_text>"
        )

    return client.chat.completions.create(
        model=mg_opts.litellm_model_name,
        response_model=PageType,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )


def collect_relevant_pages(path, min_chars=5, max_front=20, max_back=10):
    # type: (str|Path, int, int, int) -> List[Page]
    """Collect relevant content for Metadata extraction"""
    path = Path(path)
    log.debug(f"{path.name} - Scan for relevant content")
    seen = set()
    to_collect = ["TITLE_PAGE", "IMPRINT", "TABLE_OF_CONTENTS"]
    other_count = 0
    pages = []
    with fitz.open(path) as doc:
        # Scan start of book
        for page_number in range(len(doc)):
            if page_number > 20:
                log.debug(f"{path.name} - Exhausted the first {max_front} pages")
                break
            try:
                log.debug(f"{path.name} - Extracting page {page_number}")
                content = pymupdf4llm.to_markdown(doc, pages=[page_number], show_progress=False)
            except Exception as e:
                log.error(e)
                continue
            if len(content) < min_chars:
                log.debug(
                    f"{path.name} - Skip page {page_number} - less than {min_chars} chars ->"
                    f" {content[:10]}"
                )
                continue
            try:
                log.debug(f"{path.name} - Classifying page {page_number}")
                page_type = get_page_type(content, pageno=page_number)
                log.debug(f"{path.name} - Page {page_number} -> Type {page_type.page_type}")
            except Exception as e:
                log.error(e)
                continue
            if page_type.page_type in to_collect:
                seen.add(page_type.page_type)
                pages.append(Page(page_type=page_type, content=content))
            if page_type.page_type == "OTHER":
                other_count += 1
                if other_count > 8:
                    break
        if "IMPRINT" in seen:
            return pages
        # Search backward for IMPRINT
        log.debug(f"{path.name} - Scan backwarads for Imprint")
        for page_number in list(reversed(range(len(doc))))[:max_back]:
            if "IMPRINT" in seen:
                return pages
            try:
                content = pymupdf4llm.to_markdown(doc, pages=[page_number], show_progress=False)
            except Exception as e:
                log.error(e)
                continue
            if len(content) < min_chars:
                log.debug(
                    f"{path.name} - Skip page {page_number} - less than {min_chars} chars ->"
                    f" {content[:10]}"
                )
                continue
            try:
                page_type = get_page_type(content, pageno=page_number)
                log.debug(f"{path.name} - Page {page_number} -> Type {page_type.page_type}")
            except Exception as e:
                log.error(e)
                continue
            if page_type.page_type in to_collect:
                seen.add(page_type.page_type)
                pages.append(Page(page_type=page_type, content=content))
    return pages


if __name__ == "__main__":
    from rich import print

    here = Path(__file__).parent
    pages = collect_relevant_pages(here.parent / ".data/test1.pdf")
    print(pages)

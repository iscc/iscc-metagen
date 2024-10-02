import pymupdf
import pymupdf4llm
from loguru import logger as log
from pathlib import Path
from pymupdf import Document
import io


def pdf_open(doc):
    # type: (str|Path|Document) -> Document
    """
    Open a PDF file or return an already opened PDF Document object.

    :param doc: File path as string or Path, or an open Document object
    :return: An open Document object
    :raises TypeError: If input is not a string, Path, or Document object
    """
    if isinstance(doc, (str, Path)):
        doc = pymupdf.open(doc)
    elif not isinstance(doc, Document):
        raise TypeError("Input must be a string, Path, or Document object")
    return doc


def pdf_extract_pages(doc, first=8, middle=0, last=3):
    # type: (str|Path|Document, int, int, int) -> str
    """Extract relevant pages as a single Markdown text"""
    doc = pdf_open(doc)
    log.debug(f"{doc.name} -> {doc.page_count}")
    first = list(range(first)) if first else []
    center = doc.page_count // 2
    middle = list(range(center, center + middle)) if middle else []
    last = list(range(doc.page_count - last, doc.page_count)) if last else []
    page_numbers = first + middle + last
    return pymupdf4llm.to_markdown(
        doc,
        pages=page_numbers,
        embed_images=False,
        page_chunks=False,
        show_progress=False,
    )


def pdf_extract_cover(doc):
    # type: (str|Path|Document) -> io.BytesIO|None
    """
    Extract the first page of a PDF as a cover image.

    :param doc: File path as string or Path, or an open Document object
    :return: An in-memory image object (BytesIO) or None if extraction fails
    """
    doc = pdf_open(doc)
    try:
        first_page = doc[0]
        pix = first_page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # Scale up for better quality
        img_bytes = pix.tobytes("png")
        return io.BytesIO(img_bytes)
    except Exception as e:
        log.error(f"Failed to extract cover image: {e}")
        return None


if __name__ == "__main__":
    here = Path(__file__).parent
    text = pdf_extract_pages(here.parent / ".data/test1.pdf")
    print(text)

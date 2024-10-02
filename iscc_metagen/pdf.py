import pymupdf
import pymupdf4llm
from loguru import logger as log
from pathlib import Path
from pymupdf import Document
import io
from PIL import Image


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
    Extract the first page of a PDF as a cover image, resized if necessary.

    :param doc: File path as string or Path, or an open Document object
    :return: An in-memory image object (BytesIO) or None if extraction fails
    """
    doc = pdf_open(doc)
    try:
        first_page = doc[0]
        pix = first_page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # Scale up for better quality
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Resize the image if it's larger than 1536x1536
        if img.width > 1536 or img.height > 1536:
            img.thumbnail((1536, 1536))

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        return img_byte_arr
    except Exception as e:
        log.error(f"Failed to extract cover image: {e}")
        return None


if __name__ == "__main__":
    here = Path(__file__).parent
    text = pdf_extract_pages(here.parent / ".data/test1.pdf")
    print(text)

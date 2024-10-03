import pymupdf
import pymupdf4llm
from loguru import logger as log
from pathlib import Path
from pymupdf import Document
import io
from PIL import Image
from iscc_metagen.settings import mg_opts


def pdf_open(doc):
    # type: (str|Path|Document) -> Document
    """
    Open a PDF file or return an already opened PDF Document object.

    :param doc: File path as string or Path, or an open Document object
    :return: An open Document object
    :raises TypeError: If input is not a string, Path, or Document object
    """
    if isinstance(doc, (str, Path)):
        doc = Path(doc)
        filename = doc.name
        doc = pymupdf.open(doc)
        doc.filename = filename
        log.info("Opened file with PDF processor")
    elif not isinstance(doc, Document):
        raise TypeError("Input must be a string, Path, or Document object")
    return doc


def pdf_extract_pages(doc, first=None, middle=None, last=None):
    # type: (str|Path|Document, int|None, int|None, int|None) -> str
    """
    Extract relevant pages as a single Markdown text

    :param doc: PDF document to extract pages from
    :param first: Number of pages to extract from the front (overrides settings if provided)
    :param middle: Number of pages to extract from the middle (overrides settings if provided)
    :param last: Number of pages to extract from the back (overrides settings if provided)
    :return: Extracted pages as Markdown text
    """
    doc = pdf_open(doc)

    first = first if first is not None else mg_opts.front_pages
    middle = middle if middle is not None else mg_opts.mid_pages
    last = last if last is not None else mg_opts.back_pages
    log.info(f"Extracting markdown for {first} first, {middle} middle, {last} last pages")

    first_pages = list(range(first)) if first else []
    center = doc.page_count // 2
    middle_pages = list(range(center, center + middle)) if middle else []
    last_pages = list(range(doc.page_count - last, doc.page_count)) if last else []
    page_numbers = first_pages + middle_pages + last_pages

    text_md = pymupdf4llm.to_markdown(
        doc,
        pages=page_numbers,
        embed_images=False,
        page_chunks=False,
        show_progress=False,
    )

    log.info(f"Extraced {len(text_md)} characters")
    return text_md


def pdf_extract_cover(doc):
    # type: (str|Path|Document) -> io.BytesIO|None
    """
    Extract the first page of a PDF as a cover image, resized if necessary.

    :param doc: File path as string or Path, or an open Document object
    :return: An in-memory image object (BytesIO) or None if extraction fails
    """
    doc = pdf_open(doc)
    log.info(f"Extracting cover image")
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

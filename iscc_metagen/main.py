from pathlib import Path
from litellm.types.utils import ModelResponse
from iscc_metagen.schema import BookMetadata
from iscc_metagen.client import client
from iscc_metagen.settings import mg_opts
from iscc_metagen.pdf import pdf_extract_pages
from loguru import logger


def generate(file, model=None, max_retries=None):
    # type: (str|Path, str|None, int|None) -> BookMetadata
    """
    Generate metadata from a PDF file.

    :param file: Path to the PDF file.
    :param model: AI model to use for generation.
    :param max_retries: Maximum number of retries for API calls.
    :return: Generated book metadata.
    """
    logger.info(f"Starting metadata generation for {file}")
    text = pdf_extract_pages(file)
    logger.info("Text extraction completed, generating metadata")
    metadata = generate_metadata(text, model, max_retries)
    logger.info("Metadata generation completed")
    return metadata


def generate_metadata(text, model=None, max_retries=None):
    # type: (str, str|None, int|None) -> BookMetadata
    """
    Generate metadata from text input.

    :param text: Extracted text from PDF.
    :param model: AI model to use for generation.
    :param max_retries: Maximum number of retries for API calls.
    :return: Generated book metadata.
    """
    model = model or mg_opts.litellm_model_name
    max_retries = max_retries or mg_opts.max_retries
    metadata, model_response = client.chat.completions.create_with_completion(
        model=model,
        max_retries=max_retries,
        messages=[
            {
                "role": "user",
                "content": text,
            },
        ],
        response_model=BookMetadata,
    )
    model_response: ModelResponse
    metadata.model = model_response["model"]
    response_cost = model_response._hidden_params["response_cost"]
    if response_cost is not None:
        metadata.response_cost = response_cost
    return metadata


if __name__ == "__main__":
    from rich import print

    HERE = Path(__file__).parent.absolute()
    pdf = HERE.parent / ".data/test1.pdf"
    meta = generate(pdf)
    print(meta)

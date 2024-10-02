from pathlib import Path
from litellm.types.utils import ModelResponse
from iscc_metagen.schema import BookMetadata
from iscc_metagen.client import client
from iscc_metagen.settings import mg_opts
from iscc_metagen.pdf import pdf_extract_pages


def generate(file):
    # type: (str|Path) -> BookMetadata
    text = pdf_extract_pages(file)
    return generate_metadata(text)


def generate_metadata(text, model=None, max_retries=None):
    # type: (str, str|None, int|None) -> BookMetadata
    """Generate metadata from text input"""
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
    metadata.response_cost = model_response._hidden_params["response_cost"]
    return metadata


if __name__ == "__main__":
    from rich import print

    HERE = Path(__file__).parent.absolute()
    pdf = HERE.parent / ".data/test1.pdf"
    meta = generate(pdf)
    print(meta)

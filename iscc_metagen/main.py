from pathlib import Path
from iscc_metagen.schema import BookMetadata
from iscc_metagen.client import client
from iscc_metagen.settings import opts
from iscc_metagen.pdf import pdf_extract_pages


def generate(file):
    # type: (str|Path) -> BookMetadata
    text = pdf_extract_pages(file)
    return generate_metadata(text)


def generate_metadata(text, model=opts.litellm_model_name):
    # type: (str, str) -> BookMetadata
    """Generate metadata from text input"""

    return client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": text,
            },
        ],
        response_model=BookMetadata,
    )


if __name__ == "__main__":
    from rich import print

    HERE = Path(__file__).parent.absolute()
    pdf = HERE.parent / ".data/test1.pdf"
    print(generate(pdf))

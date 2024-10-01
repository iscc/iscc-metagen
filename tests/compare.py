import fitz
import ollama
from loguru import logger as log
from iscc_metagen.main import extract_pages, generate_metadata


def compare(text):
    for entry in ollama.list()["models"]:
        name = f"ollama/{entry['name']}"
        params = entry["details"]["parameter_size"]
        quant = entry["details"]["quantization_level"]
        log.debug(f"Testing {name} - {params} - {quant}")
        try:
            result = generate_metadata(text, name)
            print(result)
        except Exception as e:
            log.error(f"Failed {name} -> {e}")


if __name__ == '__main__':
    pdf = r"C:\Users\titusz\Productions\Bachem\2021_06\bergische-streifzuege_v01_bmks.pdf"
    with fitz.open(pdf) as file:
        text = extract_pages(file, first=10, last=5)

    compare()

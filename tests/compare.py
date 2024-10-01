import fitz
import ollama
from loguru import logger as log
from iscc_metagen.main import extract_pages, generate_metadata
from rich import print
from rich.table import Table
from rich.console import Console


ignore = {
    "command-r:35b-v0.1-q3_K_S",
    "aya:35b-23-q3_K_S",
    "nextfire/paraphrase-multilingual-minilm:latest",
    "mxbai-embed-large:latest",
    "nomic-embed-text:latest",
}


def compare(text):
    # type: (str) -> None
    """
    Compare metadata generation across different Ollama models.

    :param text: Input text for metadata generation.
    """
    models = ollama.list()["models"]
    sorted_models = sorted(
        models,
        key=lambda x: (
            x["details"]["parameter_size"],
            x["details"]["quantization_level"],
        ),
    )

    results = []

    for entry in sorted_models:
        name = entry["name"]
        if name in ignore:
            log.debug(f"Skipping ignored model: {name}")
            continue

        full_name = f"ollama/{name}"
        params = entry["details"]["parameter_size"]
        quant = entry["details"]["quantization_level"]
        log.debug(f"Testing {full_name} - {params} - {quant}")
        try:
            result = generate_metadata(text, full_name)
            print(result)
            results.append((full_name, params, quant, "Success"))
        except Exception as e:
            log.error(f"Failed {full_name} -> {e}")
            results.append((full_name, params, quant, "Failed"))

    # Create and display the summary table
    table = Table(title="Model Comparison Results")
    table.add_column("Model", style="cyan")
    table.add_column("Parameters", style="magenta")
    table.add_column("Quantization", style="green")
    table.add_column("Status", style="yellow")

    for result in results:
        table.add_row(*result)

    console = Console()
    console.print(table)


if __name__ == "__main__":
    pdf = (
        r"C:\Users\titusz\Productions\Bachem\2021_06\bergische-streifzuege_v01_bmks.pdf"
    )
    with fitz.open(pdf) as file:
        text = extract_pages(file, first=10, last=5)
    compare(text)

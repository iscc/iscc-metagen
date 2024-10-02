import cyclopts
from pathlib import Path
from rich import print as rprint
from rich.json import JSON
from iscc_metagen.main import generate

app = cyclopts.App()


@app.default
def main(file_path):
    # type: (Path) -> None
    """
    Generate metadata for a given PDF file and output as rich-formatted JSON.

    :param file_path: Path to the PDF file
    """
    try:
        metadata = generate(file_path)
        json_output = metadata.model_dump_json(indent=2)
        rprint(JSON(json_output))
    except Exception as e:
        rprint(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    app()

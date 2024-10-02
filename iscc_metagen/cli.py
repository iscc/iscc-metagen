import cyclopts
from pathlib import Path
from litellm.utils import get_valid_models
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


@app.command()
def gui():
    # type: () -> None
    """
    Start the Streamlit GUI for the metadata generator.
    """
    import streamlit.web.cli as stcli
    import sys
    from pathlib import Path

    gui_file = Path(__file__).parent / "gui.py"
    sys.argv = ["streamlit", "run", str(gui_file)]
    sys.exit(stcli.main())


@app.command()
def models():
    # type: () -> None
    """
    List all LiteLLM supported models.
    """
    model_list = get_valid_models()
    rprint("[bold]Supported LiteLLM Models:[/bold]")
    for model in model_list:
        rprint(f"- {model}")


if __name__ == "__main__":
    app()

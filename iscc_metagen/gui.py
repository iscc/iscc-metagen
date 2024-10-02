import streamlit as st
from pathlib import Path
import tempfile
from iscc_metagen.main import generate
from iscc_metagen.schema import BookMetadata
from iscc_metagen.pdf import pdf_extract_cover
from iscc_metagen.settings import mg_opts


def create_sidebar():
    # type: () -> str
    """Create a sidebar with model selection."""
    with st.sidebar:
        st.header("Settings")
        default_model = mg_opts.litellm_model_name
        model_options = mg_opts.litellm_models
        model = st.selectbox(
            "Choose a model for generation",
            options=model_options,
            index=model_options.index(default_model) if default_model in model_options else 0,
            help="Select the AI model to use for metadata generation.",
        )
    return model


def format_response_cost(cost):
    # type: (float) -> str
    """Format the response cost for display."""
    return f"${cost:.4f}"


def set_page_container_style():
    # type: () -> None
    """Set max-width of the content area."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_metadata(metadata):
    # type: (BookMetadata) -> None
    """Display BookMetadata in a visually appealing manner."""
    st.markdown(
        f"""
        <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;text-align:center;">
            <h3 style="margin:0;">Response Cost: {format_response_cost(metadata.response_cost)}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header(metadata.title, divider=True)
    if metadata.subtitle:
        st.subheader(metadata.subtitle)

    st.markdown(f"**Description:** {metadata.description}")

    keywords_html = """
    <style>
    .keyword-pill {
        display: inline-block;
        color: white;
        background-color: #ff4b4b;
        border-radius: 16px;
        padding: 4px 10px;
        margin: 4px;
        font-size: 14px;
    }
    </style>
    <div>
    """
    for keyword in metadata.keywords:
        keywords_html += f'<span class="keyword-pill">{keyword}</span>'
    keywords_html += "</div>"

    st.markdown(keywords_html, unsafe_allow_html=True)

    # Create a table for the remaining metadata
    table_data = [
        ["Publisher", metadata.publisher or "N/A"],
        ["Language", metadata.language],
        ["Year Published", str(metadata.year_published) if metadata.year_published else "N/A"],
        [
            "Publisher Website",
            (
                f"<a href='{metadata.publisher_website}'>{metadata.publisher_website}</a>"
                if metadata.publisher_website
                else "N/A"
            ),
        ],
    ]

    # Add contributors to the table
    if metadata.contributors:
        contributors = ", ".join([f"{c.name} ({c.role})" for c in metadata.contributors])
        table_data.append(["Contributors", contributors])

    # Add ISBNs to the table
    if metadata.isbns:
        isbns = ", ".join(
            [f"{isbn.isbn} (Edition: {isbn.edition or 'N/A'})" for isbn in metadata.isbns]
        )
        table_data.append(["ISBNs", isbns])

    # Create custom HTML table with improved styling
    table_html = """
    <style>
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        color: var(--text-color);
    }
    .custom-table td {
        border: 1px solid var(--secondary-background-color);
        padding: 8px;
    }
    .custom-table tr:nth-child(even) {
        background-color: var(--secondary-background-color);
    }
    .custom-table tr:nth-child(odd) {
        background-color: var(--background-color);
    }
    .custom-table td:first-child {
        font-weight: bold;
        width: 30%;
    }
    </style>
    <table class="custom-table">
    """
    for row in table_data:
        table_html += f"<tr><td>{row[0]}</td><td>{row[1]}</td></tr>"
    table_html += "</table>"
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(table_html, unsafe_allow_html=True)

    # Add collapsible JSON area
    with st.expander("View JSON"):
        json_output = metadata.model_dump_json(indent=2)
        st.code(json_output, language="json")


def main():
    # type: () -> None
    st.set_page_config(page_title="MetaGen", layout="wide")
    set_page_container_style()

    # Add the sidebar and get the selected model
    selected_model = create_sidebar()

    st.title("MetaGen - Metadata Generator")
    st.subheader("Generative Structured Digital Content Metadata Recognition and Extraction")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = Path(tmp_file.name)

        # Create two columns for cover and metadata
        with st.container():
            col1, col2 = st.columns([1, 2])

            with col1:
                # Extract and display cover image
                with st.spinner("Extracting cover image..."):
                    cover_image = pdf_extract_cover(tmp_file_path)
                    if cover_image:
                        st.image(cover_image, caption="Cover Image", use_column_width=True)
                    else:
                        st.warning("Failed to extract cover image.")

            with col2:
                # Generate metadata
                with st.spinner("Generating metadata..."):
                    try:
                        # Pass the selected model to the generate function
                        metadata = generate(tmp_file_path, model=selected_model)
                        display_metadata(metadata)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                    finally:
                        tmp_file_path.unlink()  # Delete the temporary file


if __name__ == "__main__":
    main()

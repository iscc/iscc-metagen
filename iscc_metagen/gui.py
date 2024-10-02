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

    st.markdown("**Keywords:**")
    st.write(", ".join(metadata.keywords))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Publisher:** {metadata.publisher or 'N/A'}")
        st.markdown(f"**Language:** {metadata.language}")
    with col2:
        st.markdown(f"**Year Published:** {metadata.year_published or 'N/A'}")
        if metadata.publisher_website:
            st.markdown(
                "**Publisher Website:**"
                f" [{metadata.publisher_website}]({metadata.publisher_website})"
            )

    if metadata.contributors:
        st.markdown("**Contributors:**")
        for contributor in metadata.contributors:
            st.markdown(f"- {contributor.name} ({contributor.role})")

    if metadata.isbns:
        st.markdown("**ISBNs:**")
        for isbn in metadata.isbns:
            st.markdown(f"- {isbn.isbn} (Edition: {isbn.edition or 'N/A'})")
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

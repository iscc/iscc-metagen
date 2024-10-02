import streamlit as st
from pathlib import Path
import tempfile
from iscc_metagen.main import generate
from iscc_metagen.schema import BookMetadata
from iscc_metagen.pdf import pdf_extract_cover

# Set page config as the first Streamlit command
st.set_page_config(page_title="MetaGen", layout="wide")

# Add custom CSS for mobile responsiveness
st.markdown(
    """
<style>
@media (max-width: 600px) {
    .stHorizontalBlock {
        flex-wrap: wrap;
    }
    .stHorizontalBlock > div {
        width: 100% !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


def display_metadata(metadata):
    # type: (BookMetadata) -> None
    """Display BookMetadata in a visually appealing manner."""
    st.header(metadata.title)
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
    st.title("MetaGen")
    st.subheader("Book Metadata Generator")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = Path(tmp_file.name)

        # Create two columns for cover and metadata
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
            with st.spinner("Extracting metadata..."):
                try:
                    metadata = generate(tmp_file_path)
                    display_metadata(metadata)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    tmp_file_path.unlink()  # Delete the temporary file


if __name__ == "__main__":
    main()

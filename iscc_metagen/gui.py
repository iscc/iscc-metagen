import streamlit as st
from pathlib import Path
import tempfile
from iscc_metagen.main import generate
from iscc_metagen.schema import BookMetadata


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


def main():
    # type: () -> None
    st.set_page_config(page_title="ISCC MetaGen", layout="centered")
    st.title("ISCC MetaGen - Book Metadata Extractor")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = Path(tmp_file.name)

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

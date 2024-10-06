"""Vector Database"""

from loguru import logger as log
import pathlib
from qdrant_client import QdrantClient
from iscc_metagen.utils import timer
from iscc_metagen.settings import mg_opts

HERE = pathlib.Path(__file__).parent.absolute()

#
# vdb = QdrantClient(path=HERE.as_posix())
#
# with timer(f"Initialize fastembed model {mg_opts.fastembed_model_name}"):
#     vdb.set_model(mg_opts.fastembed_model_name)


def list_embedding_models():
    from rich import print as p
    from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

    p(TextEmbedding.list_supported_models())
    for entry in TextEmbedding.list_supported_models():
        p(f"{entry['model']} - {entry['size_in_GB']}")
    p("#" * 32)
    for entry in SparseTextEmbedding.list_supported_models():
        p(f"{entry['model']} - {entry['size_in_GB']}")
    p("#" * 32)
    for entry in LateInteractionTextEmbedding.list_supported_models():
        p(f"{entry['model']} - {entry['size_in_GB']}")


if __name__ == "__main__":
    from rich import print

    #
    # book_description = (
    #     "Ruby on Rails Tutorial\n"
    #     "Learn Web Development with Rails\n"
    # )
    # print(vdb.query("thema", book_description))
    # with timer("Listing models in "):
    #     list_embedding_models()
    #     print(f"Current model {vdb.embedding_model_name}")

from os import path, remove
from logging import getLogger
from torch import softmax, cuda, max, matmul
print("imported torch.(softmax, cuda, max, matmul)")
from numpy import float32
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint

from open_webui.env import SRC_LOG_LEVELS

from open_webui.retrieval.models.base_reranker import BaseReranker

log = getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class ColBERT(BaseReranker):
    def __init__(self, name, **kwargs) -> None:
        log.info("ColBERT: Loading model", name)
        self.device = "cuda" if cuda.is_available() else "cpu"

        DOCKER = kwargs.get("env") == "docker"
        if DOCKER:
            # This is a workaround for the issue with the docker container
            # where the torch extension is not loaded properly
            # and the following error is thrown:
            # /root/.cache/torch_extensions/py311_cpu/segmented_maxsim_cpp/segmented_maxsim_cpp.so: cannot open shared object file: No such file or directory

            lock_file = (
                "/root/.cache/torch_extensions/py311_cpu/segmented_maxsim_cpp/lock"
            )
            if path.exists(lock_file):
                remove(lock_file)

        self.ckpt = Checkpoint(
            name,
            colbert_config=ColBERTConfig(model_name=name),
        ).to(self.device)
        pass

    def calculate_similarity_scores(self, query_embeddings, document_embeddings):

        query_embeddings = query_embeddings.to(self.device)
        document_embeddings = document_embeddings.to(self.device)

        # Validate dimensions to ensure compatibility
        if query_embeddings.dim() != 3:
            raise ValueError(
                f"Expected query embeddings to have 3 dimensions, but got {query_embeddings.dim()}."
            )
        if document_embeddings.dim() != 3:
            raise ValueError(
                f"Expected document embeddings to have 3 dimensions, but got {document_embeddings.dim()}."
            )
        if query_embeddings.size(0) not in [1, document_embeddings.size(0)]:
            raise ValueError(
                "There should be either one query or queries equal to the number of documents."
            )

        # Transpose the query embeddings to align for matrix multiplication
        transposed_query_embeddings = query_embeddings.permute(0, 2, 1)
        # Compute similarity scores using batch matrix multiplication
        computed_scores = matmul(document_embeddings, transposed_query_embeddings)
        # Apply max pooling to extract the highest semantic similarity across each document's sequence
        maximum_scores = max(computed_scores, dim=1).values

        # Sum up the maximum scores across features to get the overall document relevance scores
        final_scores = maximum_scores.sum(dim=1)

        normalized_scores = softmax(final_scores, dim=0)

        return normalized_scores.detach().cpu().numpy().astype(float32)

    def predict(self, sentences):

        query = sentences[0][0]
        docs = [i[1] for i in sentences]

        # Embedding the documents
        embedded_docs = self.ckpt.docFromText(docs, bsize=32)[0]
        # Embedding the queries
        embedded_queries = self.ckpt.queryFromText([query], bsize=32)
        embedded_query = embedded_queries[0]

        # Calculate retrieval scores for the query against all documents
        scores = self.calculate_similarity_scores(
            embedded_query.unsqueeze(0), embedded_docs
        )

        return scores

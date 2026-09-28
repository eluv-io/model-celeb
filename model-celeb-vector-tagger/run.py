import os

import setproctitle
from dacite import from_dict

from common_ml.tagging.run_helpers import catch_errors, get_params, run_default

from celeb.ground_truth import GroundTruthFetcher
from celeb_vector_tagger.config import RuntimeConfig
from celeb_vector_tagger.model import CelebVectorTagger
from celeb_vector_tagger.pool import CelebPool
from celeb_vector_tagger.vectorstore import VectorstoreClient
from config import config


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} environment variable not set")
    return value


if __name__ == '__main__':
    setproctitle.setproctitle('model-celeb-vector-tagger')

    catch_errors()

    params = from_dict(RuntimeConfig, data=get_params())

    # the index holding model-celeb-vector's embeddings, and the content being tagged (set by the tagger)
    index_qid = _require_env("ELV_INDEX_QID")
    content_qid = _require_env("ELV_CONTENT")
    token = _require_env("ELV_TOKEN")

    fetcher = GroundTruthFetcher(url=config["ground_truth"]["url"], token=token, base_dir=config["container"]["gt_path"])
    pool_path = fetcher.fetch(params.ground_truth)

    pool = CelebPool(pool_path)

    vectorstore_url = os.getenv("ELV_VECTORSTORE_URL") or config["vectorstore"]["url"]
    vectorstore = VectorstoreClient(vectorstore_url, index_qid=index_qid, token=token)

    model = CelebVectorTagger(pool, vectorstore, content_qid=content_qid, cfg=params)

    run_default(model)

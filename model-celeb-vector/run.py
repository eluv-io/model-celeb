from dacite import from_dict
import setproctitle

from common_ml.tagging.run_helpers import catch_errors, get_params, run_default

from src.config import load_config
from src.vector import CelebVectorizer, RuntimeConfig

if __name__ == '__main__':
    setproctitle.setproctitle('model-celeb-vector')

    catch_errors()

    config = load_config()
    params = from_dict(RuntimeConfig, data=get_params())

    # detects & embeds faces only; celeb-vector-tagger names them from the vectorstore
    model = CelebVectorizer(model_input_path=config["container"]["model_path"], cfg=params)

    run_default(model)

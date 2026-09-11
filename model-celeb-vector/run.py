from dacite import from_dict
import setproctitle

from common_ml.tagging.run_helpers import catch_errors, get_params, run_default

from celeb_vector.model import CelebVectorizer
from celeb_vector.config import RuntimeConfig
from config import config

if __name__ == '__main__':
    setproctitle.setproctitle('model-celeb-vector')

    catch_errors()

    params = get_params()
    params = from_dict(RuntimeConfig, data=params)

    # no ground-truth pool compared to model-celeb/run.py
    # this container only detects & embeds faces
    model = CelebVectorizer(model_input_path=config["container"]["model_path"], cfg=params)

    run_default(model)

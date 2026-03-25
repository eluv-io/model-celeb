
from dacite import from_dict
import setproctitle

from common_ml.tagging.run_helpers import catch_errors, get_params, run_default

from celeb.model import CelebRecognition
from celeb.config import RuntimeConfig
from config import config

if __name__ == '__main__':
    setproctitle.setproctitle('model-celeb')

    catch_errors()

    params = get_params()

    params = from_dict(RuntimeConfig, data=params)

    model = CelebRecognition(model_input_path=config["container"]["model_path"], cfg=params)

    run_default(model, fps=params.fps, continue_on_error=params.continue_on_error)
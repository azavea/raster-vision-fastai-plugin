import logging

from rastervision.rv_config import RVConfig

# TODO Setup a logger for each plugin in RV main repo and get rid of this
def setup_plugin_logger(root_name):
    plugin_logger = logging.getLogger(root_name)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s:%(name)s: %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    sh.setFormatter(formatter)
    plugin_logger.addHandler(sh)
    plugin_logger.setLevel(RVConfig.get_instance().verbosity)


setup_plugin_logger('fastai_plugin')
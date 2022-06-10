import logging
logger = logging.getLogger('base')


def create_model(opt):
    # video super-resolution
    from .video_base_model import VideoBaseModel as M

    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

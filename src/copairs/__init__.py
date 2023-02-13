'''
Package to create pairwise lists based on groupby and diffby criteria
'''
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
SFORMAT = '%(levelname)s:%(asctime)s:%(name)s:%(message)s'
formatter = logging.Formatter(SFORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)

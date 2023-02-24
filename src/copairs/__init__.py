'''
Package to create pairwise lists based on sameby and diffby criteria
'''
import logging

logger = logging.getLogger('copairs')
handler = logging.StreamHandler()
SFORMAT = '%(levelname)s:%(asctime)s:%(name)s:%(message)s'
formatter = logging.Formatter(SFORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)

from .matching import Matcher, MatcherMultilabel

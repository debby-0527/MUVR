import enum

from .fivr import FIVR
from .dns import DnS
from .evve import EVVE
from .cc_web_video import CC_WEB_VIDEO
from .vcdb import VCDB
from .m2vr import M2VR
from .cuvr import CuVR

from .generators import *


class EvaluationDataset(enum.Enum):
    FIVR_5K = enum.auto()
    FIVR_200K = enum.auto()
    DNS_100K = enum.auto()
    CC_WEB_VIDEO = enum.auto()
    EVVE = enum.auto()
    VCDB = enum.auto()
    M2VR = enum.auto()
    CUVR = enum.auto()

    def get_dataset(self, distractors=False):
        return self._get_config(self, distractors)

    def get_eval_fn(self, input_type):
        return self._get_eval_fn(self, input_type)

    def _get_config(self, value, distractors=False):
        return {
            self.FIVR_5K: FIVR(version='5k'),
            self.FIVR_200K: FIVR(version='200k'),
            self.DNS_100K: DnS(),
            self.CC_WEB_VIDEO: CC_WEB_VIDEO(),
            self.EVVE: EVVE(),
            self.VCDB: VCDB(distractors),
            self.M2VR: M2VR(),
            self.CUVR: CuVR(),
        }[value]

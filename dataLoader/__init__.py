from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .tnt_blender import TNTDataset
from .dtu import DTU


dataset_dict = {'blender': BlenderDataset,
               'llff':LLFFDataset,
               'tankstemple':TanksTempleDataset,
               'tankstemple2':TNTDataset,
               'nsvf':NSVF,
               'dtu':DTU}
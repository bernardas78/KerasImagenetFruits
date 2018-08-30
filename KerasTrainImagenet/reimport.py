import importlib

# initial import. Calling '' from python command line will import these modules
# , but they will only be reloaded from files by calling re() command
# To run:
#   cd c:\labs\KerasTrainMnist\KerasTrainImagenet
#   python
#   exec(open("reimport.py").read())

from DataGen import DataGen_v1_150x150_1frame as dg_v1 
from DataGen import DataGen_v2_150x150_shift_horflip as dg_v2

from Model import Model_v1_simple1 as m_v1
from Model import Model_v2_addDropout as m_v2
from Model import Model_v3_inception as m_v3
from Model import Model_v4_inception_trainbase as m_v4

from Training import Train_v1_simple1 as t_v1
from Training import Train_v2_addDropout as t_v2
from Training import Train_v22_shifthorflip as t_v22
from Training import Train_v3_inception as t_v3
from Training import Train_v4_inception_trainbase as t_v4

from Evaluation import Eval_v1_simple as e_v1
from Evaluation import Eval_v2_top5accuracy as e_v2

import visualPred as vp

def re():
    importlib.reload(dg_v1)
    importlib.reload(dg_v2)

    importlib.reload(m_v1)
    importlib.reload(m_v2)
    importlib.reload(m_v3)
    importlib.reload(m_v4)

    importlib.reload(t_v1)
    importlib.reload(t_v2)
    importlib.reload(t_v22)
    importlib.reload(t_v3)
    importlib.reload(t_v4)

    importlib.reload(e_v1)
    importlib.reload(e_v2)

    importlib.reload(vp)

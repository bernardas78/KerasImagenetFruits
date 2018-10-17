import importlib

# initial import. Calling '' from python command line will import these modules
# , but they will only be reloaded from files by calling re() command
# To run:
#   cd C:\labs\KerasImagenetFruits\KerasTrainImagenet
#   python
#   exec(open("reimport.py").read())
from keras.models import load_model

from DataGen import AugSequence as as_v1
from DataGen import DataGen_v1_150x150_1frame as dg_v1 
from DataGen import DataGen_v2_150x150_shift_horflip as dg_v2

from Model import Model_v1_simple1 as m_v1
from Model import Model_v2_addDropout as m_v2
from Model import Model_v3_inception as m_v3
from Model import Model_v4_inception_trainbase as m_v4
from Model import Model_v5_3cnn as m_v5
from Model import Model_v6_4cnn as m_v6
from Model import Model_v7_5cnn as m_v7

from Training import Train_v1_simple1 as t_v1
from Training import Train_v2_addDropout as t_v2
from Training import Train_v22_shifthorflip as t_v22
from Training import Train_v3_inception as t_v3
from Training import Train_v4_inception_trainbase as t_v4
from Training import Train_v5_8x8shifts as t_v5
from Training import Train_v6_12x12shifts as t_v6
from Training import Train_v7_8x8shifts_dropout as t_v7
from Training import Train_v8_12x12shifts_dropout as t_v8
from Training import Train_v20_224x224_size as t_v20
from Training import Train_v21_cropsize_12to16 as t_v21
from Training import Train_v23_L1_size_3_to_7 as t_v23
from Training import Train_v26_L1_stride1to2_filters32to96_maxpool as t_v26
from Training import Train_v30_L2_size1to5_stride1to2_filters32to256_maxpsize2to3 as t_v30
from Training import Train_v31_addL3 as t_v31
from Training import Train_v32_Dense128to4096 as t_v32
from Training import Train_v33_addL4 as t_v33
from Training import Train_v34_addDense2 as t_v34
from Training import Train_v35_addL5 as t_v35
from Training import Train_v36_rmDropoutAfterCnn as t_v36
from Training import Train_v37_dense4096to128 as t_v37
from Training import Train_v38_useFitGen as t_v38

from Evaluation import Eval_v1_simple as e_v1
from Evaluation import Eval_v2_top5accuracy as e_v2

import visualPred as vp

def re():
    importlib.reload(as_v1)
    importlib.reload(dg_v1)
    importlib.reload(dg_v2)

    importlib.reload(m_v1)
    importlib.reload(m_v2)
    importlib.reload(m_v3)
    importlib.reload(m_v4)
    importlib.reload(m_v5)
    importlib.reload(m_v6)
    importlib.reload(m_v7)

    importlib.reload(t_v1)
    importlib.reload(t_v2)
    importlib.reload(t_v22)
    importlib.reload(t_v3)
    importlib.reload(t_v4)
    importlib.reload(t_v5)
    importlib.reload(t_v6)
    importlib.reload(t_v7)
    importlib.reload(t_v8)
    importlib.reload(t_v20)
    importlib.reload(t_v21)
    importlib.reload(t_v23)
    importlib.reload(t_v26)
    importlib.reload(t_v30)
    importlib.reload(t_v31)
    importlib.reload(t_v32)
    importlib.reload(t_v33)
    importlib.reload(t_v34)
    importlib.reload(t_v35)
    importlib.reload(t_v36)
    importlib.reload(t_v37)
    importlib.reload(t_v38)

    importlib.reload(e_v1)
    importlib.reload(e_v2)

    importlib.reload(vp)

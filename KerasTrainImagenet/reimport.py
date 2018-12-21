import importlib

# initial import. Calling '' from python command line will import these modules
# , but they will only be reloaded from files by calling re() command
# To run:
#   cd C:\labs\KerasImagenetFruits\KerasTrainImagenet
#   python
#   exec(open("reimport.py").read())
from keras.models import load_model

from DataGen import AugSequence as as_v1
from DataGen import AugSequence_v2_Threaded as as_v2
from DataGen import AugSequence_v3_randomcrops as as_v3
from DataGen import AugSequence_v4_PcaDistortion as as_v4
from DataGen import AugSequence_v5_vggPreprocess as as_v5
from DataGen import DataGen_v1_150x150_1frame as dg_v1 
from DataGen import DataGen_v2_150x150_shift_horflip as dg_v2

from Model import Model_v1_simple1 as m_v1
from Model import Model_v2_addDropout as m_v2
from Model import Model_v3_inception as m_v3
from Model import Model_v4_inception_trainbase as m_v4
from Model import Model_v5_3cnn as m_v5
from Model import Model_v6_4cnn as m_v6
from Model import Model_v7_5cnn as m_v7
from Model import Model_v8_sgd as m_v8
from Model import Model_v9_noDropout as m_v9
from Model import Model_v10_vgg as m_v10
from Model import Model_v11_pretVggPlusSoftmax as m_v11

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
from Training import Train_v39_crops12to1 as t_v39
from Training import Train_v40_ilsvrc14data as t_v40
from Training import Train_v41_Threaded as t_v41
from Training import Train_v42_randomcrops as t_v42
from Training import Train_v43_eval5frames as t_v43
from Training import Train_v44_earlyStopping as t_v44
from Training import Train_v45_eval10frames as t_v45
from Training import Train_v46_20to50classes as t_v46
from Training import Train_v47_kryzhevski as t_v47
from Training import Train_v48_adamToSgd as t_v48
from Training import Train_v49_noDrouput as t_v49
from Training import Train_v50_experimetalDropout as t_v50
from Training import Train_v51_optimumDropout as t_v51
from Training import Train_v52_centeredInput as t_v52
from Training import Train_v53_PcaDistortion as t_v53
from Training import Train_v54_50to100classes as t_v54
from Training import Train_v55_vggSmallFilters as t_v55
from Training import Train_v56_boundingBoxes as t_v56
from Training import Train_v57_vggPreTrained as t_v57
from Training import Train_v58_preProcessFromVgg as t_v58
from Training import Train_v59_vggCutLastLayer as t_v59
from Evaluation import Eval_v1_simple as e_v1
from Evaluation import Eval_v2_top5accuracy as e_v2
from Evaluation import Eval_v3_5framesaccuracy as e_v3
from Evaluation import Eval_v4_10framesaccuracy as e_v4

import visualPred as vp
import visualPredImagenet as vpi

def re():
    importlib.reload(as_v1)
    importlib.reload(as_v2)
    importlib.reload(as_v3)
    importlib.reload(as_v4)
    importlib.reload(as_v5)
    importlib.reload(dg_v1)
    importlib.reload(dg_v2)

    importlib.reload(m_v1)
    importlib.reload(m_v2)
    importlib.reload(m_v3)
    importlib.reload(m_v4)
    importlib.reload(m_v5)
    importlib.reload(m_v6)
    importlib.reload(m_v7)
    importlib.reload(m_v8)
    importlib.reload(m_v9)
    importlib.reload(m_v10)
    importlib.reload(m_v11)

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
    importlib.reload(t_v39)
    importlib.reload(t_v40)
    importlib.reload(t_v41)
    importlib.reload(t_v42)
    importlib.reload(t_v43)
    importlib.reload(t_v44)
    importlib.reload(t_v45)
    importlib.reload(t_v46)
    importlib.reload(t_v47)
    importlib.reload(t_v48)
    importlib.reload(t_v49)
    importlib.reload(t_v50)
    importlib.reload(t_v51)
    importlib.reload(t_v52)
    importlib.reload(t_v53)
    importlib.reload(t_v54)
    importlib.reload(t_v55)
    importlib.reload(t_v56)
    importlib.reload(t_v57)
    importlib.reload(t_v58)
    importlib.reload(t_v59)

    importlib.reload(e_v1)
    importlib.reload(e_v2)
    importlib.reload(e_v3)
    importlib.reload(e_v4)
    
    importlib.reload(vp)
    importlib.reload(vpi)

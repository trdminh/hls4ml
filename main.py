from Py2C import Py2C
import tensorflow as tf
pyc_lib = Py2C(model_path="model-nodropout.h5",
               torch=False,
               input_size=(9,128),
               type="fxp",
               fxp_para=(32, 16),
               num_of_output=1,
               choose_only_output=True,
               ide="vs")
pyc_lib.convert2C()
pyc_lib.WriteCfile()
pyc_lib.Write_Float_Weights_File()


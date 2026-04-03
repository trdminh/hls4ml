#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Padding_Conv1D_0(fxp input_Pad_Conv[276], fxp output_Pad_Conv[552]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Conv1D_0(fxp Input_Conv[552],fxp Output_Conv[64], fxp bias[32], fxp kernel[13248]);
void Padding_Conv1D_1(fxp input_Pad_Conv[32], fxp output_Pad_Conv[96]);
void Conv1D_1(fxp Input_Conv[96],fxp Output_Conv[8], fxp bias[8], fxp kernel[768]);

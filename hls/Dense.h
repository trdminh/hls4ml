#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Dense_0(fxp input_Dense[8],fxp output_Dense[128],fxp bias[128],fxp weight[1024]);
void Dense_1(fxp input_Dense[128],fxp output_Dense[64],fxp bias[64],fxp weight[8192]);
void Dense_2(fxp input_Dense[64],fxp &output_Dense_0,fxp bias[6],fxp weight[384]);

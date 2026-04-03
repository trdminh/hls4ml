#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Padding_Pool_0(fxp input_Pad_Pool[64], fxp output_Pad_Pool[64]);
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Max_Pool1D_0(fxp input_MaxPooling[64], fxp output_MaxPooling[32]);
void Padding_Pool_1(fxp input_Pad_Pool[8], fxp output_Pad_Pool[16]);
void Max_Pool1D_1(fxp input_MaxPooling[16], fxp output_MaxPooling[8]);
void flatten0(fxp input_Flatten[8],fxp output_Flatten[8]);

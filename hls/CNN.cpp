#include "Conv.h"
#include "Pool.h"
#include "Dense.h"
#include <algorithm>
#include <string.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void CNN(fxp InModel[276],fxp &OutModel0,fxp Weights[23854]){
	fxp reshape[276];
	fxp OutPadConv0[552];
	fxp conv1d[64];
	fxp OutPadPool0[64];
	fxp max_pooling1d[32];
	fxp OutPadConv1[96];
	fxp conv1d_1[8];
	fxp OutPadPool1[16];
	fxp max_pooling1d_1[8];
	fxp flatten[8];
	fxp dense[128];
	fxp dense_1[64];
	Padding_Conv1D_0(conv1d,OutPadConv0);
	Conv1D_0(conv1d,conv1d,&Weights[13248],&Weights[0]);
	Padding_Pool_0(max_pooling1d,OutPadPool0);
	Max_Pool1D_0(max_pooling1d,max_pooling1d);
	Padding_Conv1D_1(conv1d_1,OutPadConv1);
	Conv1D_1(conv1d_1,conv1d_1,&Weights[14048],&Weights[13280]);
	Padding_Pool_1(max_pooling1d_1,OutPadPool1);
	Max_Pool1D_1(max_pooling1d_1,max_pooling1d_1);
	flatten0(flatten,flatten);
	Dense_0(dense,dense,&Weights[15080],&Weights[14056]);
	Dense_1(dense_1,dense_1,&Weights[23400],&Weights[15208]);
	Dense_2(OutModel0,OutModel0,&Weights[23848],&Weights[23464]);
}

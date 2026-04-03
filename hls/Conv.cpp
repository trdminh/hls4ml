#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Padding_Conv1D_0(fxp input_Pad_Conv[276], fxp output_Pad_Conv[552]){
	loop_for_3_channel_pad_0:
	for (int c = 0; c < 138; c++){
		loop_for_channel_pad_0:
		for (int n = 0; n < 4; n++){
			if (n < 1 || n >= 3) output_Pad_Conv[4 * c + n]=0; else output_Pad_Conv[4 * c + n] = input_Pad_Conv[2 * c + n - 1];
		}
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Conv1D_0(fxp Input_Conv[552],fxp Output_Conv[64], fxp bias[32], fxp kernel[13248]){
	loop_for_channel_0:
	int stride = 1;
	for (int n = 0; n < 32; n++){
		loop_for_ap_0:
		for (int y = 0; y < 2; y++){
			fxp s = 0;
			loop_for_fc_0:
			for (int k = 0; k < 138; k++){
				loop_for_fa_0:
				for (int j = 0; j < 3; j++){
					s=s+(kernel[138*3*n+3*k+j])*(Input_Conv[4*k+j+y*stride]);}
			}
			if ((s+bias[n])<0) Output_Conv[2*n+y]=0; else Output_Conv[2*n+y]=s+bias[n];
		}
	}
}
void Padding_Conv1D_1(fxp input_Pad_Conv[32], fxp output_Pad_Conv[96]){
	loop_for_3_channel_pad_1:
	for (int c = 0; c < 32; c++){
		loop_for_channel_pad_1:
		for (int n = 0; n < 3; n++){
			if (n < 1 || n >= 2) output_Pad_Conv[3 * c + n]=0; else output_Pad_Conv[3 * c + n] = input_Pad_Conv[1 * c + n - 1];
		}
	}
}
void Conv1D_1(fxp Input_Conv[96],fxp Output_Conv[8], fxp bias[8], fxp kernel[768]){
	loop_for_channel_1:
	int stride = 1;
	for (int n = 0; n < 8; n++){
		loop_for_ap_1:
		for (int y = 0; y < 1; y++){
			fxp s = 0;
			loop_for_fc_1:
			for (int k = 0; k < 32; k++){
				loop_for_fa_1:
				for (int j = 0; j < 3; j++){
					s=s+(kernel[32*3*n+3*k+j])*(Input_Conv[3*k+j+y*stride]);}
			}
			if ((s+bias[n])<0) Output_Conv[1*n+y]=0; else Output_Conv[1*n+y]=s+bias[n];
		}
	}
}

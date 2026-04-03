#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Dense_0(fxp input_Dense[8],fxp output_Dense[128],fxp bias[128],fxp weight[1024]){
	fxp out_Dense[128];
	loop_for_a_Dense_0:
	for (int i = 0; i < 128; i++){
		fxp s=0;
		loop_for_b_Dense_0:
		for (int j = 0; j < 8; j++){
			s+=input_Dense[j]*weight[j*128+i];
		}
		out_Dense[i]=s+bias[i];
	}
	for (int i = 0; i < 128; i++){
		if (out_Dense[i] < 0) output_Dense[i] = 0; else output_Dense[i] = out_Dense[i];
	}
}
void Dense_1(fxp input_Dense[128],fxp output_Dense[64],fxp bias[64],fxp weight[8192]){
	fxp out_Dense[64];
	loop_for_a_Dense_1:
	for (int i = 0; i < 64; i++){
		fxp s=0;
		loop_for_b_Dense_1:
		for (int j = 0; j < 128; j++){
			s+=input_Dense[j]*weight[j*64+i];
		}
		out_Dense[i]=s+bias[i];
	}
	for (int i = 0; i < 64; i++){
		if (out_Dense[i] < 0) output_Dense[i] = 0; else output_Dense[i] = out_Dense[i];
	}
}
#include <hls_math.h>
void Dense_2(fxp input_Dense[64],fxp &output_Dense_0,fxp bias[6],fxp weight[384]){
	fxp out_Dense[6];
	loop_for_a_Dense_2:
	for (int i = 0; i < 6; i++){
		fxp s=0;
		loop_for_b_Dense_2:
		for (int j = 0; j < 64; j++){
			s+=input_Dense[j]*weight[j*6+i];
		}
		out_Dense[i]=s+bias[i];
	}
	int maxindex = 0;
	fxp max=out_Dense[0];
	loop_detect:
	for (int i=0; i<6; i++){
		if (out_Dense[i]> max) {
			max=out_Dense[i];
			maxindex=i;
		}
	}
	fxp sum_exp_x = 0.0;
	for(int i = 0; i <6;i++){
		sum_exp_x += hls::exp(out_Dense[i]- out_Dense[maxindex]);
	}
	fxp max_value = out_Dense[maxindex];
	for(int i = 0; i <6;i++){
		out_Dense[i] = hls::exp(out_Dense[i] - max_value) / sum_exp_x;
	}
	fxp maxindex_2 = 0;
	fxp max_2 = out_Dense[0];
	for(int i = 0; i <6;i++){
		if (out_Dense[i] > max_2) {
			max_2 = out_Dense[i];
			maxindex_2 = i;
		}
	}
	output_Dense0 = maxindex_2;
}

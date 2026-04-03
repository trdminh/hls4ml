#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Padding_Pool_0(fxp input_Pad_Pool[64], fxp output_Pad_Pool[64]){
	loop_for_3_channel_pad_0:
	for (int c = 0; c < 32; c++){
		loop_for_channel_pad_0:
		for (int n = 0; n < 2; n++){
			if (n < 0 || n >= 2) output_Pad_Pool[2 * c + n]=0; else output_Pad_Pool[2 * c + n] = input_Pad_Pool[2 * c + n - 0];
		}
	}
}
#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Max_Pool1D_0(fxp input_MaxPooling[64], fxp output_MaxPooling[32]){
	int PoolSize = 2;
	int stride= 2;
	int index = 0;
	loop_for_channel_pool_0:
	for (int z = 0; z < 32; z++){
		index = 0;
		loop_for_weight_pool_0:
		for (int y = 0; y < 1; y++){
			fxp max = -10;
			for (int j = 0; j < PoolSize; j++)
			{
				int pool_index = 2 * z + j + y * stride;
				fxp pool_value = input_MaxPooling[pool_index];
				if (pool_value > max) max=pool_value;
			}
			int out_index = 1 * z + index;
			output_MaxPooling[out_index]=max;
			index++;
		}
	}
}
void Padding_Pool_1(fxp input_Pad_Pool[8], fxp output_Pad_Pool[16]){
	loop_for_3_channel_pad_1:
	for (int c = 0; c < 8; c++){
		loop_for_channel_pad_1:
		for (int n = 0; n < 2; n++){
			if (n >= 2) output_Pad_Pool[2 * c + n]=0; else output_Pad_Pool[2 * c + n] = input_Pad_Pool[1 * c + n];
		}
	}
}
void Max_Pool1D_1(fxp input_MaxPooling[16], fxp output_MaxPooling[8]){
	int PoolSize = 2;
	int stride= 2;
	int index = 0;
	loop_for_channel_pool_1:
	for (int z = 0; z < 8; z++){
		index = 0;
		loop_for_weight_pool_1:
		for (int y = 0; y < 1; y++){
			fxp max = -10;
			for (int j = 0; j < PoolSize; j++)
			{
				int pool_index = 2 * z + j + y * stride;
				fxp pool_value = input_MaxPooling[pool_index];
				if (pool_value > max) max=pool_value;
			}
			int out_index = 1 * z + index;
			output_MaxPooling[out_index]=max;
			index++;
		}
	}
}
void flatten0(fxp input_Flatten[8],fxp output_Flatten[8]){
	int hs = 0;
	loop_for_a_flatten:
	for (int i = 0; i < 8; i++){
		loop_for_c_flatten:
		for (int j = 0; j < 1; j++){
			output_Flatten[hs] = input_Flatten[1*i+j];
			hs++;
		}
	}
}

#define _CRT_SECURE_NO_WARNINGS
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <fstream>
#include <iostream>
#include "CNN.h"
#include "Conv.h"
#include "Pool.h"
#include "Dense.h"
#define NumberOfPicture ...
#define d ...
int main(){
	fxp OutModel0;
	fxp* Weights = (fxp*)malloc(23854 * sizeof(fxp));
	float tmp;
	FILE* Weight = fopen("Float_Weights.txt", "r");
	for (int i = 0; i < 23854; i++){
		fscanf(Weight, "%f", &tmp);
		*(Weights + i)=tmp;
	}
	fclose(Weight);
	////read Input
	fxp* InModel = (fxp*)malloc((NumberOfPicture * d * 276) * sizeof(fxp));
	FILE* Input = fopen("X.txt", "r");
	for (int i = 0; i < NumberOfPicture * d * 276; i++){
		fscanf(Input, "%f", &tmp);
		*(InModel + i)=tmp;
	}
	fclose(Input);
	//Read Label
	fxp*Label = (fxp*)malloc((NumberOfPicture) * sizeof(fxp));
	FILE* Output = fopen("Y.txt", "r");
	for (int i = 0; i < NumberOfPicture ; i++)
	{
		fscanf(Output, "%f", &tmp);
		*(Label + i) = tmp;
	}
	fclose(Output);
	fxp OutArray[NumberOfPicture] = {};
	fxp Image[d * 276] = {};
	for (int i = 0; i < NumberOfPicture ; i++)
	{
		int startIndex = i * d * 276;
		for (int k = 0; k < d * 276; k++)
		{
			Image[k] = *(InModel + startIndex + k);
		}
		CNN(Image, OutModel0, Weights);
		OutArray[i] = OutModel;
	}
	float countTrue = 0;
	for (int i = 0; i < NumberOfPicture; i++)
	{
		int labelValue = *(Label + i);
		if (labelValue == OutArray[i])
		{
			countTrue = countTrue + 1;
		}
	}
	float accuracy = (float)((countTrue / NumberOfPicture) * 100);
	std::cout << "accuracy of Model: " << accuracy << "%\n";
	//std::cout << "Result: " <<  OutModel <<  "\n";
	return 0;
}

//----- MLX SENSORS START -----//
#define IndexSensorMap_pr k_IndexSensorMap_pr
//----- MLX SENSORS END -----//

#ifdef METALCOMPUTE
kernel void SensorsKernel(
    const device unsigned int *p_CONSTANT_BUFFER_UINT [[buffer(0)]],
    const device unsigned int *p_INDEX_MEX [[buffer(1)]],
    const device unsigned int *p_INDEX_UINT [[buffer(2)]],
    const device unsigned int *p_UINT_BUFFER [[buffer(3)]],
    device mexType *p_MEX_BUFFER_0 [[buffer(4)]],
    device mexType *p_MEX_BUFFER_1 [[buffer(5)]],
    device mexType *p_MEX_BUFFER_2 [[buffer(6)]],
    device mexType *p_MEX_BUFFER_3 [[buffer(7)]],
    device mexType *p_MEX_BUFFER_4 [[buffer(8)]],
    device mexType *p_MEX_BUFFER_5 [[buffer(9)]],
    device mexType *p_MEX_BUFFER_6 [[buffer(10)]],
    device mexType *p_MEX_BUFFER_7 [[buffer(11)]],
    device mexType *p_MEX_BUFFER_8 [[buffer(12)]],
    device mexType *p_MEX_BUFFER_9 [[buffer(13)]],
    device mexType *p_MEX_BUFFER_10 [[buffer(14)]],
    device mexType *p_MEX_BUFFER_11 [[buffer(15)]],
    uint gid [[thread_position_in_grid]])
{
#endif
//----- MLX SENSORS START -----//
#ifdef MLX
    uint gid = thread_position_in_grid.x;
#endif

// #ifdef MLX // MLX Copy output from last loop
// 	CurZone = 0;
// 	index_copy1 = IndN1N2(i, j, 0);
// 	index_copy2 = N1 * N2;

// 	if ((SelRMSorPeak & SEL_RMS)) // RMS was selected, and it is always at the location 0 of dim 5
// 	{
// 		if (IS_Sigmaxx_SELECTED(SelMapsRMSPeak))
// 			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmaxx) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmaxx);
// 		if (IS_Sigmayy_SELECTED(SelMapsRMSPeak))
// 			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmayy) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmayy);

// 		if (IS_Pressure_SELECTED(SelMapsRMSPeak))
// 			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Pressure) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Pressure);
// 		if (IS_Sigmaxy_SELECTED(SelMapsRMSPeak))
// 			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmaxy) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmaxy);
// 	}
// 	if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK)) // If both PEAK and RMS were selected we save in the far part of the array
// 		index_copy1 += index_copy2 * NumberSelRMSPeakMaps;
// 	if (SelRMSorPeak & SEL_PEAK)
// 	{
// 		if (IS_Sigmaxx_SELECTED(SelMapsRMSPeak))
// 			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmaxx) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmaxx);
// 		if (IS_Sigmayy_SELECTED(SelMapsRMSPeak))
// 			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmayy) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmayy);
// 		if (IS_Pressure_SELECTED(SelMapsRMSPeak))
// 			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Pressure) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Pressure);
// 		if (IS_Sigmaxy_SELECTED(SelMapsRMSPeak))
// 			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmaxy) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmaxy);
// 	}

// 	// index_copy1=(((_PT)nStep)/((_PT)SensorSubSampling)-((_PT)SensorStart))*((_PT)NumberSensors)+(_PT)gid;
// 	// _PT subarrsize=(((_PT)NumberSensors)*(((_PT)TimeSteps)/((_PT)SensorSubSampling)+1-((_PT)SensorStart)));
// 	// if (IS_Vx_SELECTED(SelMapsSensors))
// 	// 	ELD(SensorOutput,index_copy1+subarrsize*IndexSensor_Vx) = ELD_OLD(SensorOutput,index_copy1+subarrsize*IndexSensor_Vx);
// 	// if (IS_Vy_SELECTED(SelMapsSensors))
// 	// 	ELD(SensorOutput,index_copy1+subarrsize*IndexSensor_Vy) = ELD_OLD(SensorOutput,index_copy1+subarrsize*IndexSensor_Vy);
// 	// if (IS_Sigmaxx_SELECTED(SelMapsSensors))
// 	// 	ELD(SensorOutput,index_copy1+subarrsize*IndexSensor_Sigmaxx) = ELD_OLD(SensorOutput,index_copy1+subarrsize*IndexSensor_Sigmaxx);
// 	// if (IS_Sigmayy_SELECTED(SelMapsSensors))
// 	// 	ELD(SensorOutput,index_copy1+subarrsize*IndexSensor_Sigmayy) = ELD_OLD(SensorOutput,index_copy1+subarrsize*IndexSensor_Sigmayy);
// 	// if (IS_Sigmaxy_SELECTED(SelMapsSensors))
// 	// 	ELD(SensorOutput,index_copy1+subarrsize*IndexSensor_Sigmaxy) = ELD_OLD(SensorOutput,index_copy1+subarrsize*IndexSensor_Sigmaxy);
// 	// if (IS_Pressure_SELECTED(SelMapsSensors))
// 	// 	ELD(SensorOutput,index_copy1+subarrsize*IndexSensor_Pressure) = ELD_OLD(SensorOutput,index_copy1+subarrsize*IndexSensor_Pressure);
// 	// if (IS_Pressure_Gx_SELECTED(SelMapsSensors))
// 	// 	ELD(SensorOutput,index_copy1+subarrsize*IndexSensor_Pressure_gx) = ELD_OLD(SensorOutput,index_copy1+subarrsize*IndexSensor_Pressure_gx);
// 	// if (IS_Pressure_Gy_SELECTED(SelMapsSensors))
// 	// 	ELD(SensorOutput,index_copy1+subarrsize*IndexSensor_Pressure_gy) = ELD_OLD(SensorOutput,index_copy1+subarrsize*IndexSensor_Pressure_gy);
// #endif

    _PT sj = (_PT)gid;

    if (sj >= (_PT)NumberSensors)
        return;

    _PT index = (((_PT)nStep) / ((_PT)SensorSubSampling) - ((_PT)SensorStart)) * ((_PT)NumberSensors) + (_PT)sj;
    _PT i, j;
    _PT index2, index3,
        subarrsize = (((_PT)NumberSensors) * (((_PT)TimeSteps) / ((_PT)SensorSubSampling) + 1 - ((_PT)SensorStart)));
    index2 = IndexSensorMap_pr[sj] - 1;

    mexType accumX = 0.0, accumY = 0.0,
            accumXX = 0.0, accumYY = 0.0,
            accumXY = 0.0, accum_p = 0, accum_p_gx = 0, accum_p_gy = 0;
    
    for (_PT CurZone = 0; CurZone < ZoneCount; CurZone++)
    {
		EL(Sigma_xy,i,j) += 1; // ADDED
// #ifdef MLX // MLX Copy output from last loop
// 		if (IsOnPML_I(i) == 1 || IsOnPML_J(j) == 1)
// 		{
// 			EL(V_x_x, i, j) = EL_OLD(V_x_x, i, j);
// 			EL(V_y_x, i, j) = EL_OLD(V_y_x, i, j);
// 			EL(V_x_y, i, j) = EL_OLD(V_x_y, i, j);
// 			EL(V_y_y, i, j) = EL_OLD(V_y_y, i, j);
// 			EL(Sigma_x_xx, i, j) = EL_OLD(Sigma_x_xx, i, j);
// 			EL(Sigma_y_xx, i, j) = EL_OLD(Sigma_y_xx, i, j);
// 			EL(Sigma_x_yy, i, j) = EL_OLD(Sigma_x_yy, i, j);
// 			EL(Sigma_y_yy, i, j) = EL_OLD(Sigma_y_yy, i, j);
// 			EL(Sigma_x_xy, i, j) = EL_OLD(Sigma_x_xy, i, j);
// 			EL(Sigma_y_xy, i, j) = EL_OLD(Sigma_y_xy, i, j);

// 			if (i == N1 - 1)
// 			{
// 				EL(V_x_x, i + 1, j) = EL_OLD(V_x_x, i + 1, j);
// 				EL(V_y_x, i + 1, j) = EL_OLD(V_y_x, i + 1, j);
// 				EL(Sigma_x_xy, i + 1, j) = EL_OLD(Sigma_x_xy, i + 1, j);
// 				EL(Sigma_y_xy, i + 1, j) = EL_OLD(Sigma_y_xy, i + 1, j);
// 			}
// 			if (j == N2 - 1)
// 			{
// 				EL(V_x_y, i, j + 1) = EL_OLD(V_x_y, i, j + 1);
// 				EL(V_y_y, i, j + 1) = EL_OLD(V_y_y, i, j + 1);
// 				EL(Sigma_x_xy, i, j + 1) = EL_OLD(Sigma_x_xy, i, j + 1);
// 				EL(Sigma_y_xy, i, j + 1) = EL_OLD(Sigma_y_xy, i, j + 1);
// 			}
// 			if (i == N1 - 1 && j == N2 - 1)
// 			{
// 				EL(V_x_y, i + 1, j + 1) = EL_OLD(V_x_y, i + 1, j + 1);
// 				EL(V_y_y, i + 1, j + 1) = EL_OLD(V_y_y, i + 1, j + 1);
// 				EL(Sigma_x_xy, i + 1, j + 1) = EL_OLD(Sigma_x_xy, i + 1, j + 1);
// 				EL(Sigma_y_xy, i + 1, j + 1) = EL_OLD(Sigma_y_xy, i + 1, j + 1);
// 			}
// 		}

// 		EL(Vx, i, j) = EL_OLD(Vx, i, j);
// 		EL(Vy, i, j) = EL_OLD(Vy, i, j);
// 		index_copy1 = Ind_Sigma_xy(i, j);
// 		ELD(Rxy, index_copy1) = ELD_OLD(Rxy, index_copy1);
// 		EL(Sigma_xy, i, j) = EL_OLD(Sigma_xy, i, j);
// 		EL(Sigma_xx, i, j) = EL_OLD(Sigma_xx, i, j);
// 		EL(Sigma_yy, i, j) = EL_OLD(Sigma_yy, i, j);
// 		EL(Pressure, i, j) = EL_OLD(Pressure, i, j);

// 		if (i == N1 - 1)
// 		{
// 			EL(Vx, i + 1, j) = EL_OLD(Vx, i + 1, j);
// 			index_copy1 = Ind_Sigma_xy(i + 1, j);
// 			ELD(Rxy, index_copy1) = ELD_OLD(Rxy, index_copy1);
// 			EL(Sigma_xy, i + 1, j) = EL_OLD(Sigma_xy, i + 1, j);
// 		}
// 		if (j == N2 - 1)
// 		{
// 			EL(Vy, i, j + 1) = EL_OLD(Vy, i, j + 1);
// 			index_copy1 = Ind_Sigma_xy(i, j + 1);
// 			ELD(Rxy, index_copy1) = ELD_OLD(Rxy, index_copy1);
// 			EL(Sigma_xy, i, j + 1) = EL_OLD(Sigma_xy, i, j + 1);
// 		}
// 		if (i == N1 - 1 && j == N2 - 1)
// 		{
// 			index_copy1 = Ind_Sigma_xy(i + 1, j + 1);
// 			ELD(Rxy, index_copy1) = ELD_OLD(Rxy, index_copy1);
// 			EL(Sigma_xy, i + 1, j + 1) = EL_OLD(Sigma_xy, i + 1, j + 1);
// 		}

// 		index_copy1 = Ind_Sigma_xx(i, j);
// 		ELD(Rxx, index_copy1) = ELD_OLD(Rxx, index_copy1);
// 		ELD(Ryy, index_copy1) = ELD_OLD(Ryy, index_copy1);

// #endif
//         i = index2 % (N1);
//         j = index2 / N1;

//         if (IS_Vx_SELECTED(SelMapsSensors))
//             accumX += EL(Vx, i, j);
//         if (IS_Vy_SELECTED(SelMapsSensors))
//             accumY += EL(Vy, i, j);

//         index3 = Ind_Sigma_xx(i, j);
// #ifdef METAL
//         // No idea why in this kernel the ELD(SigmaXX...) macros do not expand correctly
//         // So we go a bit more manual
//         if (IS_Sigmaxx_SELECTED(SelMapsSensors))
//             accumXX += k_Sigma_xx_pr[index3];
//         if (IS_Sigmayy_SELECTED(SelMapsSensors))
//             accumYY += k_Sigma_yy_pr[index3];
//         if (IS_Pressure_SELECTED(SelMapsSensors))
//             accum_p += k_Pressure_pr[index3];
//         if (IS_Pressure_Gx_SELECTED(SelMapsSensors))
//             accum_p_gx += (k_Pressure_pr[Ind_Sigma_xx(i + 1, j)] - k_Pressure_pr[Ind_Sigma_xx(i - 1, j)]) * 0.5;
//         if (IS_Pressure_Gy_SELECTED(SelMapsSensors))
//             accum_p_gy += (k_Pressure_pr[Ind_Sigma_xx(i, j + 1)] - k_Pressure_pr[Ind_Sigma_xx(i, j - 1)]) * 0.5;
//         index3 = Ind_Sigma_xy(i, j);
//         if (IS_Sigmaxy_SELECTED(SelMapsSensors))
//             accumXY += k_Sigma_xy_pr[index3];

// #else
//         if (IS_Sigmaxx_SELECTED(SelMapsSensors))
//             accumXX += ELD(Sigma_xx, index3);
//         if (IS_Sigmayy_SELECTED(SelMapsSensors))
//             accumYY += ELD(Sigma_yy, index3);
//         if (IS_Pressure_SELECTED(SelMapsSensors))
//             accum_p += ELD(Pressure, index3);
//         if (IS_Pressure_Gx_SELECTED(SelMapsSensors))
//             accum_p_gx += (Pressure_pr[Ind_Sigma_xx(i + 1, j)] - Pressure_pr[Ind_Sigma_xx(i - 1, j)]) * 0.5;
//         if (IS_Pressure_Gy_SELECTED(SelMapsSensors))
//             accum_p_gy += (Pressure_pr[Ind_Sigma_xx(i, j + 1)] - Pressure_pr[Ind_Sigma_xx(i, j - 1)]) * 0.5;
//         index3 = Ind_Sigma_xy(i, j);
//         if (IS_Sigmaxy_SELECTED(SelMapsSensors))
//             accumXY += ELD(Sigma_xy, index3);
// #endif
    }
    
    // accumX /= ZoneCount;
    // accumY /= ZoneCount;
    // accumXX /= ZoneCount;
    // accumYY /= ZoneCount;
    // accumXY /= ZoneCount;
    // accum_p /= ZoneCount;
    
    // // ELD(SensorOutput,index)=accumX*accumX+accumY*accumY+accumZ*accumZ;
    // if (IS_Vx_SELECTED(SelMapsSensors))
    //     ELD(SensorOutput, index + subarrsize * IndexSensor_Vx) = accumX;
    // if (IS_Vy_SELECTED(SelMapsSensors))
    //     ELD(SensorOutput, index + subarrsize * IndexSensor_Vy) = accumY;
    // if (IS_Sigmaxx_SELECTED(SelMapsSensors))
    //     ELD(SensorOutput, index + subarrsize * IndexSensor_Sigmaxx) = accumXX;
    // if (IS_Sigmayy_SELECTED(SelMapsSensors))
    //     ELD(SensorOutput, index + subarrsize * IndexSensor_Sigmayy) = accumYY;
    // if (IS_Sigmaxy_SELECTED(SelMapsSensors))
    //     ELD(SensorOutput, index + subarrsize * IndexSensor_Sigmaxy) = accumXY;
    // if (IS_Pressure_SELECTED(SelMapsSensors))
    //     ELD(SensorOutput, index + subarrsize * IndexSensor_Pressure) = accum_p;
    // if (IS_Pressure_Gx_SELECTED(SelMapsSensors))
    //     ELD(SensorOutput, index + subarrsize * IndexSensor_Pressure_gx) = accum_p_gx;
    // if (IS_Pressure_Gy_SELECTED(SelMapsSensors))
    //     ELD(SensorOutput, index + subarrsize * IndexSensor_Pressure_gy) = accum_p_gy;
//----- MLX SENSORS END -----//
#ifndef MLX
}
#endif
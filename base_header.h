/***** Indexing2D *****/ 
// Comments like the one BELOW are needed for proper MLX kernel compilation
//----- MLX HEADER START -----//
#ifndef INDEXING_DEF
//----- MLX HEADER END -----//
// Comments like the one ABOVE are needed for proper MLX kernel compilation
//----- MLX HEADER START -----//
#define INDEXING_DEF

// Pointer type? used for indexes
typedef unsigned long  _PT;

// Used for interface types
typedef unsigned char interface_t;
typedef _PT tIndex ;

// Used when checking if a voxel exists at an interface
#define inside 0x00
#define frontLine 0x01
#define frontLinep1 0x02
#define frontLinep2 0x04
#define backLine   0x08
#define backLinem1 0x10
#define backLinem2 0x20

//#define USE_2ND_ORDER_EDGES 1

// USE_2ND_ORDER_EDGES macro defaults to 0 (False) for 2D sims
#ifdef USE_2ND_ORDER_EDGES

//#define REQUIRES_2ND_ORDER_P(__I) ((interface ## __I & frontLinep1)  || (interface ## __I & frontLine) || (interface ## __I & backLine) )
#define REQUIRES_2ND_ORDER_P(__I) (interface ## __I & frontLine)

//#define REQUIRES_2ND_ORDER_M(__I) ((interface ## __I & backLinem1) || (interface ## __I & frontLine) || (interface ## __I & backLine))
#define REQUIRES_2ND_ORDER_M(__I) (interface ## __I & frontLine)
#else

#define REQUIRES_2ND_ORDER_P(__I) (0)

#define REQUIRES_2ND_ORDER_M(__I) (0)
#endif

// define XOR macro
#define XOR(_a,_b) ((!(_a) && (_b)) || ((_a) && !(_b)))

// Coefficients for the 4th order FDTD
#define CA (1.1250) // 9/8
#define CB (0.0416666666666666643537020320309) // 1/24

// For METAL backend, need to CONCAT twice to ensure concatenation works
// properly when a and/or b are macros, which is the case for __PRE_MAT macro
#if defined(METAL)
#define CONCAT2(a, b) a ## b
#define CONCAT(a, b) CONCAT2(a, b)
#endif

// Matrix/array name prefix
#if defined(CUDA)
#define __PRE_MAT 
#elif defined(METAL)
#define __PRE_MAT k_ // for kernel specific data?
#else
#define __PRE_MAT
#endif

// Macros to grab array element values using i,j indices (EL) or single index for 1D array (ELD)
// when a matrix prefix is required
#if defined(METAL)
#define EL(_Mat,_i,_j) CONCAT(__PRE_MAT,_Mat ## _pr[Ind_##_Mat(_i,_j)])
#define EL_OLD(_Mat,_i,_j) CONCAT(__PRE_MAT,_Mat ## _old_pr[Ind_##_Mat(_i,_j)])
#define ELD(_Mat,_index) CONCAT(__PRE_MAT,_Mat ## _pr[_index])
#define ELD_OLD(_Mat,_index) CONCAT(__PRE_MAT,_Mat ## _old_pr[_index])
#else
#define EL(_Mat,_i,_j) __PRE_MAT _Mat##_pr[Ind_##_Mat(_i,_j)]
#define ELD(_Mat,_index) __PRE_MAT _Mat##_pr[_index]
#endif

// Macros to grab array element values using i,j indices (ELO) or single index for 1D array (ELDO)
// Simpler macro that doesn't use matrix prefixing
#define ELO(_Mat,_i,_j)  _Mat##_pr[Ind_##_Mat(_i,_j)]
#define ELDO(_Mat,_index)  _Mat##_pr[_index]
//----- MLX HEADER END -----//

// Same as above, but required for host side calculation
#define hELO(_Mat,_i,_j)  _Mat##_pr[hInd_##_Mat(_i,_j)]


//////////////////////////////////////////////
// Macros for indexing or indexing checks on host side
#define hInd_Source(a,b)((b)*INHOST(N2)+a)

#define hIndN1N2Snap(a,b) ((b)*INHOST(N1)+a)
#define hIndN1N2(a,b,_ZoneSize)   ((b)*INHOST(N1)    +a+(CurZone*(_ZoneSize)))
#define hIndN1p1N2(a,b,_ZoneSize) ( (b)*(INHOST(N1)+1)+a+(CurZone*(_ZoneSize)))
#define hIndN1N2p1(a,b,_ZoneSize) ((b)*(INHOST(N1))  +a+(CurZone*(_ZoneSize)))
#define hIndN1N2(a,b,_ZoneSize) ((b)*INHOST(N1)    +a+(CurZone*(_ZoneSize)))
#define hIndN1p1N2p1(a,b,_ZoneSize) ((b)*(INHOST(N1)+1)+a +(CurZone*(_ZoneSize)))

#define hCorrecI(_i,_j) ((_j)>hLimit_J_low_PML && (_j)<hLimit_J_up_PML && (_i)> hLimit_I_low_PML ?hSizeCorrI :0)
#define hCorrecJ(_j) ((_j)>hLimit_J_low_PML+1 ?((_j)<hLimit_J_up_PML?((_j)-hLimit_J_low_PML-1)*(hSizeCorrI):hSizeCorrI*hSizeCorrJ):0)

#define hIndexPML(_i,_j,_ZoneSize)  (hIndN1N2(_i,_j,_ZoneSize) - hCorrecI(_i,_j) - hCorrecJ(_j))

#define hIndexPMLxp1(_i,_j,_ZoneSize) (hIndN1p1N2(_i,_j,_ZoneSize) - hCorrecI(_i,_j) )
#define hIndexPMLyp1(_i,_j,_ZoneSize) (hIndN1N2p1(_i,_j,_ZoneSize) - hCorrecI(_i,_j) )

#define hInd_MaterialMap(_i,_j) (hIndN1p1N2p1(_i,_j,(INHOST(N1)+1)*(INHOST(N2)+1)))

#define hInd_V_x(_i,_j) (hIndN1p1N2(_i,_j,(INHOST(N1)+1)*INHOST(N2)))
#define hInd_V_y(_i,_j) (hIndN1N2p1(_i,_j,INHOST(N1)*(INHOST(N2)+1)))

#define hInd_Vx(_i,_j) (hIndN1p1N2(_i,_j,(INHOST(N1)+1)*INHOST(N2)))
#define hInd_Vy(_i,_j) (hIndN1N2p1(_i,_j,INHOST(N1)*(INHOST(N2)+1)))

#define hInd_Sigma_xx(_i,_j) (hIndN1N2(_i,_j,INHOST(N1)*INHOST(N2)))
#define hInd_Sigma_yy(_i,_j) (hIndN1N2(_i,_j,INHOST(N1)*INHOST(N2)))

#define hInd_Pressure(_i,_j) (hIndN1N2(_i,_j,INHOST(N1)*INHOST(N2)))
#define hInd_Pressure_old(_i,_j) (hIndN1N2(_i,_j,INHOST(N1)*INHOST(N2)))

#define hInd_Sigma_xy(_i,_j) (hIndN1p1N2p1(_i,_j,(INHOST(N1)+1)*(INHOST(N2)+1)))

#define hInd_SqrAcc(_i,_j) (hIndN1N2(_i,_j,INHOST(N1)*INHOST(N2)))

#define hInd_V_x_x(_i,_j) (hIndexPMLxp1(_i,_j,INHOST(SizePMLxp1)))
#define hInd_V_y_x(_i,_j) (hIndexPMLxp1(_i,_j,INHOST(SizePMLxp1)))

#define hInd_V_x_y(_i,_j) (hIndexPMLyp1(_i,_j,INHOST(SizePMLyp1)))
#define hInd_V_y_y(_i,_j) (hIndexPMLyp1(_i,_j,INHOST(SizePMLyp1)))



#define hInd_Sigma_x_xx(_i,_j) (hIndexPML(_i,_j,INHOST(SizePML)) )
#define hInd_Sigma_y_xx(_i,_j) (hIndexPML(_i,_j,INHOST(SizePML)) )

#define hInd_Sigma_x_yy(_i,_j) (hIndexPML(_i,_j,INHOST(SizePML)) )
#define hInd_Sigma_y_yy(_i,_j) (hIndexPML(_i,_j,INHOST(SizePML)) )

//----- MLX HEADER START -----//
// Macros to check if index is in PML region
#define IsOnPML_I(_i) ((_i) <=Limit_I_low_PML || (_i)>=Limit_I_up_PML ? 1:0)
#define IsOnPML_J(_j) ((_j) <=Limit_J_low_PML || (_j)>=Limit_J_up_PML ? 1:0)
#define IsOnPML_K(_k) ((_k) <=Limit_K_low_PML || (_k)>=Limit_K_up_PML ? 1:0)

#define IsOnLowPML_I(_i) (_i) <=Limit_I_low_PML
#define IsOnLowPML_J(_j) (_j) <=Limit_J_low_PML
#define IsOnLowPML_K(_k) (_k) <=Limit_K_low_PML

////////////////////////////////////////
// Macros for indexing or indexing checks
#define Ind_Source(a,b)((b)*N2+a)

#define IndN1N2Snap(a,b) ((b)*N1+a)

#define IndN1N2(a,b,_ZoneSize)   ((b)*N1    +a+(CurZone*(_ZoneSize)))
#define IndN1p1N2(a,b,_ZoneSize) ( (b)*(N1+1)+a+(CurZone*(_ZoneSize)))
#define IndN1N2p1(a,b,_ZoneSize) ((b)*N1 +a+(CurZone*(_ZoneSize)))
// #define IndN1N2(a,b,_ZoneSize) ((b)*N1   +a+(CurZone*(_ZoneSize)))
#define IndN1p1N2p1(a,b,_ZoneSize) ((b)*(N1+1)+a +(CurZone*(_ZoneSize)))

#define CorrecI(_i,_j) ((_j)>Limit_J_low_PML  && (_j)<Limit_J_up_PML  && (_i)> Limit_I_low_PML ?SizeCorrI :0)
#define CorrecJ(_j) ( (_j)>Limit_J_low_PML+1 ?((_j)<Limit_J_up_PML?((_j)-Limit_J_low_PML-1)*(SizeCorrI):SizeCorrI*SizeCorrJ):0)

#define IndexPML(_i,_j,_ZoneSize)  (IndN1N2(_i,_j,_ZoneSize) - CorrecI(_i,_j) - CorrecJ(_j))

#define IndexPMLxp1(_i,_j,_ZoneSize) (IndN1p1N2(_i,_j,_ZoneSize) - CorrecI(_i,_j) - CorrecJ(_j))
#define IndexPMLyp1(_i,_j,_ZoneSize) (IndN1N2p1(_i,_j,_ZoneSize) - CorrecI(_i,_j) - CorrecJ(_j))
#define IndexPMLxp1yp1(_i,_j,_ZoneSize) (IndN1p1N2p1(_i,_j,_ZoneSize) - CorrecI(_i,_j) - CorrecJ(_j))

#define Ind_MaterialMap(_i,_j) (IndN1p1N2p1(_i,_j,(N1+1)*(N2+1)))

#define Ind_V_x(_i,_j) (IndN1p1N2(_i,_j,(N1+1)*N2))
#define Ind_V_y(_i,_j) (IndN1N2p1(_i,_j,N1*(N2+1)))


#define Ind_Vx(_i,_j) (IndN1p1N2(_i,_j,(N1+1)*N2))
#define Ind_Vy(_i,_j) (IndN1N2p1(_i,_j,N1*(N2+1)))
#define Ind_Vz(_i,_j) (IndN1N2(_i,_j,N1*N2))

#define Ind_Sigma_xx(_i,_j) (IndN1N2(_i,_j,N1*N2))
#define Ind_Sigma_yy(_i,_j) (IndN1N2(_i,_j,N1*N2))
#define Ind_Sigma_zz(_i,_j) (IndN1N2(_i,_j,N1*N2))

#define Ind_Pressure(_i,_j) (IndN1N2(_i,_j,N1*N2))

#define Ind_Sigma_xy(_i,_j) (IndN1p1N2p1(_i,_j,(N1+1)*(N2+1)))
#define Ind_Sigma_xz(_i,_j) (IndN1p1N2p1(_i,_j,(N1+1)*(N2+1)))
#define Ind_Sigma_yz(_i,_j) (IndN1p1N2p1(_i,_j,(N1+1)*(N2+1)))

#define Ind_SqrAcc(_i,_j) (IndN1N2(_i,_j,N1*N2))

#define Ind_V_x_x(_i,_j) (IndexPMLxp1(_i,_j,SizePMLxp1))
#define Ind_V_y_x(_i,_j) (IndexPMLxp1(_i,_j,SizePMLxp1))

#define Ind_V_x_y(_i,_j) (IndexPMLyp1(_i,_j,SizePMLyp1))
#define Ind_V_y_y(_i,_j) (IndexPMLyp1(_i,_j,SizePMLyp1))


#define Ind_Sigma_x_xx(_i,_j) (IndexPML(_i,_j,SizePML) )
#define Ind_Sigma_y_xx(_i,_j) (IndexPML(_i,_j,SizePML) )

#define Ind_Sigma_x_yy(_i,_j) (IndexPML(_i,_j,SizePML) )
#define Ind_Sigma_y_yy(_i,_j) (IndexPML(_i,_j,SizePML) )

#define Ind_Sigma_x_xy(_i,_j)(IndexPMLxp1yp1(_i,_j,SizePMLxp1yp1) )
#define Ind_Sigma_y_xy(_i,_j)(IndexPMLxp1yp1(_i,_j,SizePMLxp1yp1) )


#define iPML(_i) ((_i) <=Limit_I_low_PML ? (_i) : ((_i)<Limit_I_up_PML ? PML_Thickness : (_i)<N1 ? PML_Thickness-1-(_i)+Limit_I_up_PML:0))
#define jPML(_j) ((_j) <=Limit_J_low_PML ? (_j) : ((_j)<Limit_J_up_PML ? PML_Thickness : (_j)<N2 ? PML_Thickness-1-(_j)+Limit_J_up_PML:0))
//----- MLX HEADER END -----//

#if defined(CUDA) || defined(OPENCL)
#define InvDXDT_I 	(IsOnLowPML_I(i) ? gpuInvDXDTpluspr[iPML(i)] : gpuInvDXDTplushppr[iPML(i)] )
#define DXDT_I 		(IsOnLowPML_I(i) ? gpuDXDTminuspr[iPML(i)] : gpuDXDTminushppr[iPML(i)] )
#define InvDXDT_J 	(IsOnLowPML_J(j) ? gpuInvDXDTpluspr[jPML(j)] : gpuInvDXDTplushppr[jPML(j)] )
#define DXDT_J 		(IsOnLowPML_J(j) ? gpuDXDTminuspr[jPML(j)] : gpuDXDTminushppr[jPML(j)] )

#define InvDXDThp_I 	(IsOnLowPML_I(i) ? gpuInvDXDTplushppr[iPML(i)] : gpuInvDXDTpluspr[iPML(i)] )
#define DXDThp_I 		(IsOnLowPML_I(i) ? gpuDXDTminushppr[iPML(i)] : gpuDXDTminuspr[iPML(i)] )
#define InvDXDThp_J 	(IsOnLowPML_J(j) ? gpuInvDXDTplushppr[jPML(j)] : gpuInvDXDTpluspr[jPML(j)] )
#define DXDThp_J 		(IsOnLowPML_J(j) ? gpuDXDTminushppr[jPML(j)] : gpuDXDTminuspr[jPML(j)] )
#else
//----- MLX HEADER START -----//
#define InvDXDT_I 	(IsOnLowPML_I(i) ? InvDXDTplus_pr[iPML(i)] : InvDXDTplushp_pr[iPML(i)] )
#define DXDT_I 		(IsOnLowPML_I(i) ? DXDTminus_pr[iPML(i)] : DXDTminushp_pr[iPML(i)] )
#define InvDXDT_J 	(IsOnLowPML_J(j) ? InvDXDTplus_pr[jPML(j)] : InvDXDTplushp_pr[jPML(j)] )
#define DXDT_J 		(IsOnLowPML_J(j) ? DXDTminus_pr[jPML(j)] : DXDTminushp_pr[jPML(j)] )

#define InvDXDThp_I 	(IsOnLowPML_I(i) ? InvDXDTplushp_pr[iPML(i)] : InvDXDTplus_pr[iPML(i)] )
#define DXDThp_I 		(IsOnLowPML_I(i) ? DXDTminushp_pr[iPML(i)] : DXDTminus_pr[iPML(i)] )
#define InvDXDThp_J 	(IsOnLowPML_J(j) ? InvDXDTplushp_pr[jPML(j)] : InvDXDTplus_pr[jPML(j)] )
#define DXDThp_J 		(IsOnLowPML_J(j) ? DXDTminushp_pr[jPML(j)] : DXDTminus_pr[jPML(j)] )
//----- MLX HEADER END -----//
#endif

//----- MLX HEADER START -----//
#define MASK_Vx   			0x0000000001
#define MASK_Vy   			0x0000000002
#define MASK_Sigmaxx    	0x0000000004
#define MASK_Sigmayy    	0x0000000008
#define MASK_Sigmaxy    	0x0000000010
#define MASK_Pressure      	0x0000000020
#define MASK_Pressure_Gx   	0x0000000040
#define MASK_Pressure_Gy   	0x0000000080

#define IS_Vx_SELECTED(_Value) 					(_Value &MASK_Vx)
#define IS_Vy_SELECTED(_Value) 					(_Value &MASK_Vy)
#define IS_Sigmaxx_SELECTED(_Value) 			(_Value &MASK_Sigmaxx)
#define IS_Sigmayy_SELECTED(_Value) 			(_Value &MASK_Sigmayy)
#define IS_Sigmaxy_SELECTED(_Value) 			(_Value &MASK_Sigmaxy)
#define IS_Pressure_SELECTED(_Value) 			(_Value &MASK_Pressure)
#define IS_Pressure_Gx_SELECTED(_Value) 		(_Value &MASK_Pressure_Gx)
#define IS_Pressure_Gy_SELECTED(_Value) 		(_Value &MASK_Pressure_Gy)

#define COUNT_SELECTIONS(_VarName,_Value) \
				{ _VarName =0;\
					_VarName += IS_Vx_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Vy_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Sigmaxx_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Sigmayy_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Sigmaxy_SELECTED(_Value) ? 1 : 0; \
					_VarName += IS_Pressure_SELECTED(_Value) ? 1 : 0;\
					_VarName += IS_Pressure_Gx_SELECTED(_Value) ? 1 : 0;\
					_VarName += IS_Pressure_Gy_SELECTED(_Value) ? 1 : 0}

#define SEL_RMS			  	0x0000000001
#define SEL_PEAK   			0x0000000002

#define ACCOUNT_RMSPEAK(_VarName)\
if IS_ ## _VarName ## _SELECTED(INHOST(SelMapsRMSPeak)) \
{\
	 INHOST(IndexRMSPeak_ ## _VarName)=curMapIndex;\
	 curMapIndex++; }

 #define ACCOUNT_SENSOR(_VarName)\
 if IS_ ## _VarName ## _SELECTED(INHOST(SelMapsSensors)) \
 {\
 	 INHOST(IndexSensor_ ## _VarName)=curMapIndex;\
 	 curMapIndex++; }
//----- MLX HEADER END -----//

#if defined(METAL)
// #ifndef METALCOMPUTE
// #define CInd_N1 0
// #define CInd_N2 1
// #define CInd_Limit_I_low_PML 2
// #define CInd_Limit_J_low_PML 3
// #define CInd_Limit_I_up_PML 4
// #define CInd_Limit_J_up_PML 5
// #define CInd_SizeCorrI 6
// #define CInd_SizeCorrJ 7
// #define CInd_PML_Thickness 8
// #define CInd_NumberSources 9
// #define CInd_NumberSensors 10
// #define CInd_TimeSteps 11
// #define CInd_SizePML 12
// #define CInd_SizePMLxp1 13
// #define CInd_SizePMLyp1 14
// #define CInd_SizePMLxp1yp1 15
// #define CInd_ZoneCount 16
// #define CInd_SelRMSorPeak 17
// #define CInd_SelMapsRMSPeak 18
// #define CInd_IndexRMSPeak_Vx 19
// #define CInd_IndexRMSPeak_Vy 20
// #define CInd_IndexRMSPeak_Sigmaxx 21
// #define CInd_IndexRMSPeak_Sigmayy 22
// #define CInd_IndexRMSPeak_Sigmaxy 23
// #define CInd_NumberSelRMSPeakMaps 24
// #define CInd_SelMapsSensors 25
// #define CInd_IndexSensor_Vx 26
// #define CInd_IndexSensor_Vy 27
// #define CInd_IndexSensor_Sigmaxx 28
// #define CInd_IndexSensor_Sigmayy 29
// #define CInd_IndexSensor_Sigmaxy 30
// #define CInd_NumberSelSensorMaps 31
// #define CInd_SensorSubSampling 32
// #define CInd_nStep 33
// #define CInd_TypeSource 34
// #define CInd_CurrSnap 35
// #define CInd_LengthSource 36
// #define CInd_IndexRMSPeak_Pressure 37
// #define CInd_IndexSensor_Pressure 38
// #define CInd_IndexSensor_Pressure_gx 39
// #define CInd_IndexSensor_Pressure_gy 40
// #define CInd_SensorStart 41


// //Make LENGTH_CONST_UINT one value larger than the last index
// #define LENGTH_CONST_UINT 42

// //Indexes for float
// #define CInd_DT 0
// #define CInd_InvDXDTplus 1
// #define CInd_DXDTminus (1+MAX_SIZE_PML)
// #define CInd_InvDXDTplushp (1+MAX_SIZE_PML*2)
// #define CInd_DXDTminushp (1+MAX_SIZE_PML*3)
// //Make LENGTH_CONST_MEX one value larger than the last index
// #define LENGTH_CONST_MEX (1+MAX_SIZE_PML*4)
// #else
//----- MLX HEADER START -----//
#define CInd_nStep 0
#define CInd_TypeSource 1
#define LENGTH_CONST_UINT 2
//----- MLX HEADER END -----//
// #endif

//----- MLX HEADER START -----//
#define CInd_V_x_x 0
#define CInd_V_y_x 1
#define CInd_V_x_y 2
#define CInd_V_y_y 3

#define CInd_Vx 4
#define CInd_Vy 5

#define CInd_Rxx 6
#define CInd_Ryy 7

#define CInd_Rxy 8

#define CInd_Sigma_x_xx 9
#define CInd_Sigma_y_xx 10
#define CInd_Sigma_x_yy 11
#define CInd_Sigma_y_yy 12

#define CInd_Sigma_x_xy 13
#define CInd_Sigma_y_xy 14

#define CInd_Sigma_xy 15

#define CInd_Sigma_xx 16
#define CInd_Sigma_yy 17

#define CInd_SourceFunctions 18

#define CInd_LambdaMiuMatOverH  19
#define CInd_LambdaMatOverH	 20
#define CInd_MiuMatOverH 21
#define CInd_TauLong 22
#define CInd_OneOverTauSigma	23
#define CInd_TauShear 24
#define CInd_InvRhoMatH	25
#define CInd_Ox 26
#define CInd_Oy 27
#define CInd_Pressure 28

#define CInd_SqrAcc 29

#define CInd_SensorOutput 30

#define LENGTH_INDEX_MEX 31

#define CInd_IndexSensorMap  0
#define CInd_SourceMap	1
#define CInd_MaterialMap 2

#define LENGTH_INDEX_UINT 3

//----- MLX HEADER END -----//
#endif

//----- MLX HEADER START -----//
#endif
//----- MLX HEADER END -----//

/***** kernelparamsMetal2D *****/ 
#ifdef METAL
// #ifndef METALCOMPUTE
// #define N1 p_CONSTANT_BUFFER_UINT[CInd_N1]
// #define N2 p_CONSTANT_BUFFER_UINT[CInd_N2]
// #define Limit_I_low_PML p_CONSTANT_BUFFER_UINT[CInd_Limit_I_low_PML]
// #define Limit_J_low_PML p_CONSTANT_BUFFER_UINT[CInd_Limit_J_low_PML]
// #define Limit_I_up_PML p_CONSTANT_BUFFER_UINT[CInd_Limit_I_up_PML]
// #define Limit_J_up_PML p_CONSTANT_BUFFER_UINT[CInd_Limit_J_up_PML]
// #define SizeCorrI p_CONSTANT_BUFFER_UINT[CInd_SizeCorrI]
// #define SizeCorrJ p_CONSTANT_BUFFER_UINT[CInd_SizeCorrJ]
// #define PML_Thickness p_CONSTANT_BUFFER_UINT[CInd_PML_Thickness]
// #define NumberSources p_CONSTANT_BUFFER_UINT[CInd_NumberSources]
// #define LengthSource p_CONSTANT_BUFFER_UINT[CInd_LengthSource]
// #define NumberSensors p_CONSTANT_BUFFER_UINT[CInd_NumberSensors]
// #define TimeSteps p_CONSTANT_BUFFER_UINT[CInd_TimeSteps]

// #define SizePML p_CONSTANT_BUFFER_UINT[CInd_SizePML]
// #define SizePMLxp1 p_CONSTANT_BUFFER_UINT[CInd_SizePMLxp1]
// #define SizePMLyp1 p_CONSTANT_BUFFER_UINT[CInd_SizePMLyp1]
// #define SizePMLxp1yp1 p_CONSTANT_BUFFER_UINT[CInd_SizePMLxp1yp1]
// #define ZoneCount p_CONSTANT_BUFFER_UINT[CInd_ZoneCount]

// #define SelRMSorPeak p_CONSTANT_BUFFER_UINT[CInd_SelRMSorPeak]
// #define SelMapsRMSPeak p_CONSTANT_BUFFER_UINT[CInd_SelMapsRMSPeak]
// #define IndexRMSPeak_Vx p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Vx]
// #define IndexRMSPeak_Vy p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Vy]
// #define IndexRMSPeak_Sigmaxx p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Sigmaxx]
// #define IndexRMSPeak_Sigmayy p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Sigmayy]
// #define IndexRMSPeak_Sigmaxy p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Sigmaxy]
// #define IndexRMSPeak_Pressure p_CONSTANT_BUFFER_UINT[CInd_IndexRMSPeak_Pressure]
// #define NumberSelRMSPeakMaps p_CONSTANT_BUFFER_UINT[CInd_NumberSelRMSPeakMaps]

// #define SelMapsSensors p_CONSTANT_BUFFER_UINT[CInd_SelMapsSensors]
// #define IndexSensor_Vx p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Vx]
// #define IndexSensor_Vy p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Vy]
// #define IndexSensor_Sigmaxx p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Sigmaxx]
// #define IndexSensor_Sigmayy p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Sigmayy]
// #define IndexSensor_Sigmaxy p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Sigmaxy]
// #define IndexSensor_Pressure p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Pressure]
// #define IndexSensor_Pressure_gx p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Pressure_gx]
// #define IndexSensor_Pressure_gy p_CONSTANT_BUFFER_UINT[CInd_IndexSensor_Pressure_gy]
// #define NumberSelSensorMaps p_CONSTANT_BUFFER_UINT[CInd_NumberSelSensorMaps]
// #define SensorSubSampling p_CONSTANT_BUFFER_UINT[CInd_SensorSubSampling]
// #define SensorStart p_CONSTANT_BUFFER_UINT[CInd_SensorStart]
// #define nStep p_CONSTANT_BUFFER_UINT[CInd_nStep]
// #define CurrSnap p_CONSTANT_BUFFER_UINT[CInd_CurrSnap]
// #define TypeSource p_CONSTANT_BUFFER_UINT[CInd_TypeSource]

// #define DT p_CONSTANT_BUFFER_MEX[CInd_DT]
// #define InvDXDTplus_pr (p_CONSTANT_BUFFER_MEX + CInd_InvDXDTplus)
// #define DXDTminus_pr (p_CONSTANT_BUFFER_MEX + CInd_DXDTminus)
// #define InvDXDTplushp_pr (p_CONSTANT_BUFFER_MEX + CInd_InvDXDTplushp)
// #define DXDTminushp_pr (p_CONSTANT_BUFFER_MEX + CInd_DXDTminushp)
// #else
// Comments like the one BELOW are needed for proper MLX kernel compilation
//----- MLX HEADER START -----//
#define nStep p_CONSTANT_BUFFER_UINT[CInd_nStep]
#define TypeSource p_CONSTANT_BUFFER_UINT[CInd_TypeSource]
//----- MLX HEADER END -----//
// Comments like the one ABOVE are needed for proper MLX kernel compilation
// #endif

//----- MLX HEADER START -----//
#define __def_MEX_VAR_0(__NameVar)  (&p_MEX_BUFFER_0[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_1(__NameVar)  (&p_MEX_BUFFER_1[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_2(__NameVar)  (&p_MEX_BUFFER_2[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_3(__NameVar)  (&p_MEX_BUFFER_3[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_4(__NameVar)  (&p_MEX_BUFFER_4[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_5(__NameVar)  (&p_MEX_BUFFER_5[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_6(__NameVar)  (&p_MEX_BUFFER_6[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_7(__NameVar)  (&p_MEX_BUFFER_7[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_8(__NameVar)  (&p_MEX_BUFFER_8[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_9(__NameVar)  (&p_MEX_BUFFER_9[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_10(__NameVar)  (&p_MEX_BUFFER_10[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_11(__NameVar)  (&p_MEX_BUFFER_11[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#ifdef MLX
#define __def_MEX_VAR_0_OLD(__NameVar)  (&p_MEX_BUFFER_0_OLD[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_1_OLD(__NameVar)  (&p_MEX_BUFFER_1_OLD[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_2_OLD(__NameVar)  (&p_MEX_BUFFER_2_OLD[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_3_OLD(__NameVar)  (&p_MEX_BUFFER_3_OLD[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_4_OLD(__NameVar)  (&p_MEX_BUFFER_4_OLD[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_5_OLD(__NameVar)  (&p_MEX_BUFFER_5_OLD[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_6_OLD(__NameVar)  (&p_MEX_BUFFER_6_OLD[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_7_OLD(__NameVar)  (&p_MEX_BUFFER_7_OLD[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_8_OLD(__NameVar)  (&p_MEX_BUFFER_8_OLD[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_9_OLD(__NameVar)  (&p_MEX_BUFFER_9_OLD[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_10_OLD(__NameVar)  (&p_MEX_BUFFER_10_OLD[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#define __def_MEX_VAR_11_OLD(__NameVar)  (&p_MEX_BUFFER_11_OLD[ ((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar *2])) | (((unsigned long) (p_INDEX_MEX[CInd_ ##__NameVar*2+1]))<<32) ]) 
#endif

#define __def_UINT_VAR(__NameVar)  (&p_UINT_BUFFER[ ((unsigned long) (p_INDEX_UINT[CInd_ ##__NameVar*2])) | (((unsigned long) (p_INDEX_UINT[CInd_ ##__NameVar*2+1]))<<32) ])

// #define __def_MEX_VAR(__NameVar)  (&p_MEX_BUFFER[ p_INDEX_MEX[CInd_ ##__NameVar ]]) 
// #define __def_UINT_VAR(__NameVar)  (&p_UINT_BUFFER[ p_INDEX_UINT[CInd_ ##__NameVar]])


#define k_V_x_x_pr  __def_MEX_VAR_0(V_x_x)
#define k_V_y_x_pr  __def_MEX_VAR_0(V_y_x)
#define k_V_x_y_pr  __def_MEX_VAR_0(V_x_y)
#define k_V_y_y_pr  __def_MEX_VAR_0(V_y_y)

#define k_Vx_pr  __def_MEX_VAR_1(Vx)
#define k_Vy_pr  __def_MEX_VAR_1(Vy)

#define k_Rxx_pr  __def_MEX_VAR_2(Rxx)
#define k_Ryy_pr  __def_MEX_VAR_2(Ryy)

#define k_Rxy_pr  __def_MEX_VAR_3(Rxy)

#define k_Sigma_x_xx_pr  __def_MEX_VAR_4(Sigma_x_xx)
#define k_Sigma_y_xx_pr  __def_MEX_VAR_4(Sigma_y_xx)
#define k_Sigma_x_yy_pr  __def_MEX_VAR_4(Sigma_x_yy)
#define k_Sigma_y_yy_pr  __def_MEX_VAR_4(Sigma_y_yy)

#define k_Sigma_x_xy_pr  __def_MEX_VAR_5(Sigma_x_xy)
#define k_Sigma_y_xy_pr  __def_MEX_VAR_5(Sigma_y_xy)

#define k_Sigma_xy_pr  __def_MEX_VAR_6(Sigma_xy)
#define k_Sigma_xx_pr  __def_MEX_VAR_6(Sigma_xx)

#define k_Sigma_yy_pr  __def_MEX_VAR_7(Sigma_yy)
#define k_Pressure_pr  __def_MEX_VAR_7(Pressure)

#define k_SourceFunctions_pr __def_MEX_VAR_8(SourceFunctions)

#define k_LambdaMiuMatOverH_pr  __def_MEX_VAR_9(LambdaMiuMatOverH)
#define k_LambdaMatOverH_pr     __def_MEX_VAR_9(LambdaMatOverH)
#define k_MiuMatOverH_pr        __def_MEX_VAR_9(MiuMatOverH)
#define k_TauLong_pr            __def_MEX_VAR_9(TauLong)
#define k_OneOverTauSigma_pr    __def_MEX_VAR_9(OneOverTauSigma)
#define k_TauShear_pr           __def_MEX_VAR_9(TauShear)
#define k_InvRhoMatH_pr         __def_MEX_VAR_9(InvRhoMatH)
#define k_Ox_pr  __def_MEX_VAR_9(Ox)
#define k_Oy_pr  __def_MEX_VAR_9(Oy)


#define k_SqrAcc_pr  __def_MEX_VAR_10(SqrAcc)

#define k_SensorOutput_pr  __def_MEX_VAR_11(SensorOutput)

#define k_IndexSensorMap_pr  __def_UINT_VAR(IndexSensorMap)
#define k_SourceMap_pr		 __def_UINT_VAR(SourceMap)
#define k_MaterialMap_pr	 __def_UINT_VAR(MaterialMap)

#ifdef MLX
#define k_V_x_x_old_pr  __def_MEX_VAR_0_OLD(V_x_x)
#define k_V_y_x_old_pr  __def_MEX_VAR_0_OLD(V_y_x)
#define k_V_x_y_old_pr  __def_MEX_VAR_0_OLD(V_x_y)
#define k_V_y_y_old_pr  __def_MEX_VAR_0_OLD(V_y_y)
#define k_Vx_old_pr  __def_MEX_VAR_1_OLD(Vx)
#define k_Vy_old_pr  __def_MEX_VAR_1_OLD(Vy)
#define k_Rxx_old_pr  __def_MEX_VAR_2_OLD(Rxx)
#define k_Ryy_old_pr  __def_MEX_VAR_2_OLD(Ryy)
#define k_Rxy_old_pr  __def_MEX_VAR_3_OLD(Rxy)
#define k_Sigma_x_xx_old_pr  __def_MEX_VAR_4_OLD(Sigma_x_xx)
#define k_Sigma_y_xx_old_pr  __def_MEX_VAR_4_OLD(Sigma_y_xx)
#define k_Sigma_x_yy_old_pr  __def_MEX_VAR_4_OLD(Sigma_x_yy)
#define k_Sigma_y_yy_old_pr  __def_MEX_VAR_4_OLD(Sigma_y_yy)
#define k_Sigma_x_xy_old_pr  __def_MEX_VAR_5_OLD(Sigma_x_xy)
#define k_Sigma_y_xy_old_pr  __def_MEX_VAR_5_OLD(Sigma_y_xy)
#define k_Sigma_xy_old_pr  __def_MEX_VAR_6_OLD(Sigma_xy)
#define k_Sigma_xx_old_pr  __def_MEX_VAR_6_OLD(Sigma_xx)
#define k_Sigma_yy_old_pr  __def_MEX_VAR_7_OLD(Sigma_yy)
#define k_Pressure_old_pr  __def_MEX_VAR_7_OLD(Pressure)
#define k_SourceFunctions_old_pr __def_MEX_VAR_8_OLD(SourceFunctions)
#define k_LambdaMiuMatOverH_old_pr  __def_MEX_VAR_9_OLD(LambdaMiuMatOverH)
#define k_LambdaMatOverH_old_pr     __def_MEX_VAR_9_OLD(LambdaMatOverH)
#define k_MiuMatOverH_old_pr        __def_MEX_VAR_9_OLD(MiuMatOverH)
#define k_TauLong_old_pr            __def_MEX_VAR_9_OLD(TauLong)
#define k_OneOverTauSigma_old_pr    __def_MEX_VAR_9_OLD(OneOverTauSigma)
#define k_TauShear_old_pr           __def_MEX_VAR_9_OLD(TauShear)
#define k_InvRhoMatH_old_pr         __def_MEX_VAR_9_OLD(InvRhoMatH)
#define k_Ox_old_pr  __def_MEX_VAR_9_OLD(Ox)
#define k_Oy_old_pr  __def_MEX_VAR_9_OLD(Oy)
#define k_SqrAcc_old_pr  __def_MEX_VAR_10_OLD(SqrAcc)
#define k_SensorOutput_old_pr  __def_MEX_VAR_11_OLD(SensorOutput)
#endif

//----- MLX HEADER END -----//


#ifdef METAL
// Comments like the one BELOW are needed for proper MLX kernel compilation
//----- MLX HEADER START -----//
#define CGID uint
//----- MLX HEADER END -----//
// Comments like the one ABOVE are needed for proper MLX kernel compilation
#else
#define CGID uint3
#endif

#ifndef METALCOMPUTE
#define METAL_PARAMS\
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],\
	const device mexType * p_CONSTANT_BUFFER_MEX [[ buffer(1) ]],\
	const device unsigned int *p_INDEX_MEX [[ buffer(2) ]],\
	const device unsigned int *p_INDEX_UINT [[ buffer(3) ]],\
	const device unsigned int *p_UINT_BUFFER [[ buffer(4) ]],\
	device mexType * p_MEX_BUFFER_0 [[ buffer(5) ]],\
	device mexType * p_MEX_BUFFER_1 [[ buffer(6) ]],\
	device mexType * p_MEX_BUFFER_2 [[ buffer(7) ]],\
	device mexType * p_MEX_BUFFER_3 [[ buffer(8) ]],\
	device mexType * p_MEX_BUFFER_4 [[ buffer(9) ]],\
	device mexType * p_MEX_BUFFER_5 [[ buffer(10) ]],\
	device mexType * p_MEX_BUFFER_6 [[ buffer(11) ]],\
	device mexType * p_MEX_BUFFER_7 [[ buffer(12) ]],\
	device mexType * p_MEX_BUFFER_8 [[ buffer(13) ]],\
	device mexType * p_MEX_BUFFER_9 [[ buffer(14) ]],\
	device mexType * p_MEX_BUFFER_10 [[ buffer(15) ]],\
	device mexType * p_MEX_BUFFER_11 [[ buffer(16) ]],\
	CGID gid[[thread_position_in_grid]])\
{
#else
#define METAL_PARAMS\
	const device unsigned int *p_CONSTANT_BUFFER_UINT [[ buffer(0) ]],\
	const device unsigned int *p_INDEX_MEX [[ buffer(1) ]],\
	const device unsigned int *p_INDEX_UINT [[ buffer(2) ]],\
	const device unsigned int *p_UINT_BUFFER [[ buffer(3) ]],\
	device mexType * p_MEX_BUFFER_0 [[ buffer(4) ]],\
	device mexType * p_MEX_BUFFER_1 [[ buffer(5) ]],\
	device mexType * p_MEX_BUFFER_2 [[ buffer(6) ]],\
	device mexType * p_MEX_BUFFER_3 [[ buffer(7) ]],\
	device mexType * p_MEX_BUFFER_4 [[ buffer(8) ]],\
	device mexType * p_MEX_BUFFER_5 [[ buffer(9) ]],\
	device mexType * p_MEX_BUFFER_6 [[ buffer(10) ]],\
	device mexType * p_MEX_BUFFER_7 [[ buffer(11) ]],\
	device mexType * p_MEX_BUFFER_8 [[ buffer(12) ]],\
	device mexType * p_MEX_BUFFER_9 [[ buffer(13) ]],\
	device mexType * p_MEX_BUFFER_10 [[ buffer(14) ]],\
	device mexType * p_MEX_BUFFER_11 [[ buffer(15) ]],\
	CGID gid[[thread_position_in_grid]])\
{
#endif
#endif

#ifdef MLX
//----- MLX HEADER START -----//

#define MLX_SQRACC_COPY																														    	\
	CurZone = 0;																																	\
	index_copy1 = IndN1N2(i, j, 0);																													\
	index_copy2 = N1 * N2;																															\
		                                                                                                                                            \
	if ((SelRMSorPeak & SEL_RMS)) /* RMS was selected, and it is always at the location 0 of dim 5 */                                               \
	{																																				\
		if (IS_Sigmaxx_SELECTED(SelMapsRMSPeak))	                                                                                                \
			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmaxx) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmaxx);      \
		if (IS_Sigmayy_SELECTED(SelMapsRMSPeak))									                                                                \
			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmayy) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmayy);		\
																																					\
		if (IS_Pressure_SELECTED(SelMapsRMSPeak))																									\
			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Pressure) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Pressure);	\
		if (IS_Sigmaxy_SELECTED(SelMapsRMSPeak))																									\
			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmaxy) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmaxy);		\
		if (IS_Vx_SELECTED(SelMapsRMSPeak))																											\
			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Vx) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Vx);				\
		if (IS_Vy_SELECTED(SelMapsRMSPeak))																											\
			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Vy) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Vy);				\
	}																																				\
	if ((SelRMSorPeak & SEL_RMS) && (SelRMSorPeak & SEL_PEAK)) /* If both PEAK and RMS were selected we save in the far part of the array */		\
		index_copy1 += index_copy2 * NumberSelRMSPeakMaps;																							\
	if (SelRMSorPeak & SEL_PEAK)																													\
	{																																				\
		if (IS_Sigmaxx_SELECTED(SelMapsRMSPeak))																									\
			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmaxx) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmaxx);		\
		if (IS_Sigmayy_SELECTED(SelMapsRMSPeak))																									\
			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmayy) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmayy);		\
		if (IS_Pressure_SELECTED(SelMapsRMSPeak))																									\
			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Pressure) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Pressure);	\
		if (IS_Sigmaxy_SELECTED(SelMapsRMSPeak))																									\
			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmaxy) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Sigmaxy);		\
		if (IS_Vx_SELECTED(SelMapsRMSPeak))																											\
			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Vx) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Vx);				\
		if (IS_Vy_SELECTED(SelMapsRMSPeak))																											\
			ELD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Vy) = ELD_OLD(SqrAcc, index_copy1 + index_copy2 * IndexRMSPeak_Vy);				\
	}																																				\
	threadgroup_barrier(metal::mem_flags::mem_threadgroup);

#define MLX_STRESS_COPY 													    \
	/* Copy the data from the old buffer to the new one */ 						\					
	/* This is needed for the MLX kernel compilation */ 						\						
	if (IsOnPML_I(i) == 1 || IsOnPML_J(j) == 1)									\
	{	                                                    					\
		EL(Sigma_x_xx, i, j) = EL_OLD(Sigma_x_xx, i, j);    					\
		EL(Sigma_y_xx, i, j) = EL_OLD(Sigma_y_xx, i, j);    					\
		EL(Sigma_x_yy, i, j) = EL_OLD(Sigma_x_yy, i, j);    					\
		EL(Sigma_y_yy, i, j) = EL_OLD(Sigma_y_yy, i, j);    					\
		EL(Sigma_x_xy, i, j) = EL_OLD(Sigma_x_xy, i, j);    					\
		EL(Sigma_y_xy, i, j) = EL_OLD(Sigma_y_xy, i, j);    					\
                                                            					\
		if (i == N1 - 1)                                    					\
		{                                                   					\
			EL(Sigma_x_xy, i + 1, j) = EL_OLD(Sigma_x_xy, i + 1, j); 			\
			EL(Sigma_y_xy, i + 1, j) = EL_OLD(Sigma_y_xy, i + 1, j); 			\
		} 																		\
		if (j == N2 - 1) 														\
		{ 																		\
			EL(Sigma_x_xy, i, j + 1) = EL_OLD(Sigma_x_xy, i, j + 1); 			\
			EL(Sigma_y_xy, i, j + 1) = EL_OLD(Sigma_y_xy, i, j + 1); 			\
		} 																		\
		if (i == N1 - 1 && j == N2 - 1) 										\
		{ 																		\
			EL(Sigma_x_xy, i + 1, j + 1) = EL_OLD(Sigma_x_xy, i + 1, j + 1); 	\
			EL(Sigma_y_xy, i + 1, j + 1) = EL_OLD(Sigma_y_xy, i + 1, j + 1); 	\
		} 																		\
	} 																			\
      																			\
	index_copy1 = Ind_Sigma_xy(i, j); 											\
	ELD(Rxy, index_copy1) = ELD_OLD(Rxy, index_copy1); 							\
	EL(Sigma_xy, i, j) = EL_OLD(Sigma_xy, i, j);								\
	EL(Sigma_xx, i, j) = EL_OLD(Sigma_xx, i, j);								\
	EL(Sigma_yy, i, j) = EL_OLD(Sigma_yy, i, j);								\
	EL(Pressure, i, j) = EL_OLD(Pressure, i, j);								\
                                                								\
	if (i == N1 - 1)                            								\
	{ 																			\
		index_copy1 = Ind_Sigma_xy(i + 1, j); 									\
		ELD(Rxy, index_copy1) = ELD_OLD(Rxy, index_copy1); 						\
		EL(Sigma_xy, i + 1, j) = EL_OLD(Sigma_xy, i + 1, j); 					\
	} 																			\
	if (j == N2 - 1) 															\
	{ 																			\
		index_copy1 = Ind_Sigma_xy(i, j + 1); 									\
		ELD(Rxy, index_copy1) = ELD_OLD(Rxy, index_copy1); 						\
		EL(Sigma_xy, i, j + 1) = EL_OLD(Sigma_xy, i, j + 1); 					\
	} 																			\
	if (i == N1 - 1 && j == N2 - 1) 											\
	{ 																			\
		index_copy1 = Ind_Sigma_xy(i + 1, j + 1); 								\
		ELD(Rxy, index_copy1) = ELD_OLD(Rxy, index_copy1); 						\
		EL(Sigma_xy, i + 1, j + 1) = EL_OLD(Sigma_xy, i + 1, j + 1); 			\
	} 																			\
	  																			\
	index_copy1 = Ind_Sigma_xx(i, j); 											\
	ELD(Rxx, index_copy1) = ELD_OLD(Rxx, index_copy1); 							\
	ELD(Ryy, index_copy1) = ELD_OLD(Ryy, index_copy1); 							\
	threadgroup_barrier(metal::mem_flags::mem_threadgroup);

#define MLX_PARTICLE_COPY	 													\
	/* Copy the data from the old buffer to the new one */ 						\					
	/* This is needed for the MLX kernel compilation */ 						\						
	if (IsOnPML_I(i) == 1 || IsOnPML_J(j) == 1)									\
	{	                                                    					\
		EL(V_x_x, i, j) = EL_OLD(V_x_x, i, j);			 						\
		EL(V_y_x, i, j) = EL_OLD(V_y_x, i, j);              					\
		EL(V_x_y, i, j) = EL_OLD(V_x_y, i, j);              					\
		EL(V_y_y, i, j) = EL_OLD(V_y_y, i, j);              					\
                                                            					\
		if (i == N1 - 1)                                    					\
		{                                                   					\
			EL(V_x_x, i + 1, j) = EL_OLD(V_x_x, i + 1, j);  					\
			EL(V_y_x, i + 1, j) = EL_OLD(V_y_x, i + 1, j);  					\
		} 																		\
		if (j == N2 - 1) 														\
		{ 																		\
			EL(V_x_y, i, j + 1) = EL_OLD(V_x_y, i, j + 1); 						\
			EL(V_y_y, i, j + 1) = EL_OLD(V_y_y, i, j + 1); 						\
		} 																		\
	} 																			\
      																			\
	EL(Vx, i, j) = EL_OLD(Vx, i, j); 											\
	EL(Vy, i, j) = EL_OLD(Vy, i, j); 											\
                                                								\
	if (i == N1 - 1)                            								\
	{ 																			\
		EL(Vx, i + 1, j) = EL_OLD(Vx, i + 1, j); 								\
	} 																			\
	if (j == N2 - 1) 															\
	{ 																			\
		EL(Vy, i, j + 1) = EL_OLD(Vy, i, j + 1); 								\
	} 																			\
	  																			\
	threadgroup_barrier(metal::mem_flags::mem_threadgroup);

#define MLX_SENSORS_COPY	 																																	\
	/* Copy the data from the old buffer to the new one */ 																										\					
	/* This is needed for the MLX kernel compilation */ 																										\						
																																								\
	_PT index_copy1;																																			\
																																								\
	for (int time_step = 0; time_step < nStep; time_step++)																										\
	{																																						\
		if (time_step % SensorSubSampling != 0)																														\
			continue;  																																				\
		index_copy1 = (((_PT)time_step) / ((_PT)SensorSubSampling) - ((_PT)SensorStart)) * ((_PT)NumberSensors) + gid;											\																																				
																																								\
		if (IS_Vx_SELECTED(SelMapsSensors))																														\
			ELD(SensorOutput, index_copy1 + subarrsize * IndexSensor_Vx) = ELD_OLD(SensorOutput, index_copy1 + subarrsize * IndexSensor_Vx);					\
		if (IS_Vy_SELECTED(SelMapsSensors))																														\
			ELD(SensorOutput, index_copy1 + subarrsize * IndexSensor_Vy) = ELD_OLD(SensorOutput, index_copy1 + subarrsize * IndexSensor_Vy);					\
		if (IS_Sigmaxx_SELECTED(SelMapsSensors))																												\
			ELD(SensorOutput, index_copy1 + subarrsize * IndexSensor_Sigmaxx) = ELD_OLD(SensorOutput, index_copy1 + subarrsize * IndexSensor_Sigmaxx);			\
		if (IS_Sigmayy_SELECTED(SelMapsSensors))																												\
			ELD(SensorOutput, index_copy1 + subarrsize * IndexSensor_Sigmayy) = ELD_OLD(SensorOutput, index_copy1 + subarrsize * IndexSensor_Sigmayy);			\
		if (IS_Sigmaxy_SELECTED(SelMapsSensors))																												\
			ELD(SensorOutput, index_copy1 + subarrsize * IndexSensor_Sigmaxy) = ELD_OLD(SensorOutput, index_copy1 + subarrsize * IndexSensor_Sigmaxy);			\
		if (IS_Pressure_SELECTED(SelMapsSensors))																												\
			ELD(SensorOutput, index_copy1 + subarrsize * IndexSensor_Pressure) = ELD_OLD(SensorOutput, index_copy1 + subarrsize * IndexSensor_Pressure);		\
		if (IS_Pressure_Gx_SELECTED(SelMapsSensors))																											\
			ELD(SensorOutput, index_copy1 + subarrsize * IndexSensor_Pressure_gx) = ELD_OLD(SensorOutput, index_copy1 + subarrsize * IndexSensor_Pressure_gx);	\
		if (IS_Pressure_Gy_SELECTED(SelMapsSensors))																											\
			ELD(SensorOutput, index_copy1 + subarrsize * IndexSensor_Pressure_gy) = ELD_OLD(SensorOutput, index_copy1 + subarrsize * IndexSensor_Pressure_gy);	\
	}																																							\
		threadgroup_barrier(metal::mem_flags::mem_threadgroup);
//----- MLX HEADER END -----//
#endif
/// PMLS

//----- MLX HEADER START -----//
#define IndexSensorMap_pr k_IndexSensorMap_pr
//----- MLX HEADER END -----//

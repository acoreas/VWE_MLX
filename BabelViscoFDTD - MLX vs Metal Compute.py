# Base
from math import ceil
import gc
import os
os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '1'
os.environ['METAL_DEBUG_ERROR_MODE'] = '0'
from pathlib import Path
import pickle
import platform
import re
import sys
import time

# Installed
from matplotlib import pyplot as plt
import metalcomputebabel as mc
import mlx.core as mx
import numpy as np
from scipy.optimize import fmin_slsqp
from scipy.interpolate import interp1d
from skimage.metrics import mean_squared_error, normalized_root_mse

# Local
import viscoelastic_utils as vwe_utils
from vwe import VWE

# Global Variables
global base_header, stress_kernel, particle_kernel, sensors_kernel
base_header = ""
stress_kernel = ""
particle_kernel = ""
sensors_kernel = ""
            
def main():
    #%% PARAMETERS
    # Device Specific
    gpu_device = 'M1'               # GPU device name

    # Simulation
    dt = 1.5e-7                     # time step
    medium_SOS = 1500               # m/s - water
    medium_density = 1000           # kg/m3
    pml_thickness = 12              # grid points for perfect matching layer
    points_per_wavelength = 9       # minimum step of 6 is recommended
    reflection_limit = 1.0000e-05   # reflection parameter for PML
    tx_diameter = 0.03              # m - circular piston
    tx_plane_loc = 0.01             # m - in XY plane at Z = 0.01 m
    us_amplitude = 100e3            # Pa
    us_frequency = 350e3            # Hz
    x_dim = 0.05                    # m
    y_dim = 0.10                    # m



    #%% SIMULATION DOMAIN SETUP
    # Domain Properties
    shortest_wavelength = medium_SOS/us_frequency
    spatial_step = shortest_wavelength/ points_per_wavelength

    # Domain Dimensions
    domain_dims =  np.array([x_dim,y_dim])  # in m, x,y,z
    N1 = int(np.ceil(domain_dims[0]/spatial_step)+2*pml_thickness)
    N2 = int(np.ceil(domain_dims[1]/spatial_step)+2*pml_thickness)
    print('Domain size',N1,N2)

    # Time Dimensions
    sim_time=np.sqrt(domain_dims[0]**2+domain_dims[1]**2)/medium_SOS #time to cross one corner to another
    sensor_steps=int((1/us_frequency/8)/dt) # for the sensors, we do not need so much high temporal resolution, so we are keeping 8 time points per perioid

    # Material Map
    material_map=np.zeros((N1,N2),np.uint32) # note the 32 bit size
    material_list=np.zeros((1,5)) # one material in this examples
    material_list[0,0]=medium_density # water density
    material_list[0,1]=medium_SOS # water SoS
    # all other parameters are set to 0 

    # Constants
    StaggeredConstants={}
    StaggeredConstants['ColDensity']=0
    StaggeredConstants['ColLongSOS']=1
    StaggeredConstants['ColShearSOS']=2
    StaggeredConstants['ColLongAtt']=3
    StaggeredConstants['ColShearAtt']=4
    
    
    
    #%% GENERATE SOURCE MAP + SOURCE TIME SIGNAL
    # Generate line source
    def MakeLineSource(DimX,SpatialStep,Diameter):
        # simple defintion of a circular source centred in the domain
        XDim=np.arange(DimX)*SpatialStep
        XDim-=XDim.mean()
        MaskSource=np.abs(XDim)<=(Diameter/2.0)
        return (MaskSource*1.0).astype(np.uint32)

    line_source=MakeLineSource(N1,spatial_step,tx_diameter)
    source_map=np.zeros((N1,N2),np.uint32)
    z_loc=int(np.round(tx_plane_loc/spatial_step))+pml_thickness
    source_map[:,z_loc] = line_source 

    # Plot source map
    # plt.figure()
    # plt.imshow(source_map.T)


    amp_displacement = us_amplitude/medium_density/medium_SOS # We use a 100 kPa source, we just need to convert to particle displacement
    Ox=np.zeros((N1,N2))
    Oy=np.zeros((N1,N2))
    Oy[source_map>0]=1 #only Y has a value of 1

    # Generate source time signal
    source_length = 4.0/us_frequency # we will use 4 pulses
    source_time_vector=np.arange(0,source_length+dt,dt)

    # Plot source time signal
    pulse_source = np.sin(2*np.pi*us_frequency*source_time_vector)
    # plt.figure()
    # plt.plot(source_time_vector*1e6,pulse_source)
    # plt.title('4-pulse signal')

    # note we need expressively to arrange the data in a 2D array
    pulse_source=np.reshape(pulse_source,(1,len(source_time_vector))) 
    print("Number of time points in source signal:",len(source_time_vector))
    
    
    
    #%% GENERATE SENSOR MAP
    # Define sensor map
    sensor_map=np.zeros((N1,N2),np.uint32)
    sensor_map[pml_thickness:-pml_thickness,pml_thickness:-pml_thickness]=1

    # Plot sensor map
    # plt.figure()
    # plt.imshow(sensor_map.T,cmap=plt.cm.gray)
    # plt.title('Sensor map location')   
    
    
    
    #%% VISCOELASTICE WAVE EQUATION KERNEL CODES
    global base_header,stress_kernel,particle_kernel,sensors_kernel
    with open('base_header.h','r') as f:
        base_header = f.read()
        
    with open('stress_kernel.c','r') as f:
        stress_kernel = f.read()
        
    with open('particle_kernel.c','r') as f:
        particle_kernel = f.read()
        
    with open('sensors_kernel.c','r') as f:
        sensors_kernel = f.read()
        
        
    #%% VISCOELEASTIC WAVE EQUATION PRE KERNEL CALCULATIONS  
    vwe_mc = VWE()
    vwe_mlx = VWE()

    # VWE setup
    input_params_mc, post_kernel_args_mc = vwe_utils.VWE_preparation(MaterialMap=material_map,
                                                MaterialProperties=material_list,
                                                Frequency=us_frequency,
                                                SourceMap=source_map,
                                                SourceFunctions=pulse_source,
                                                SpatialStep=spatial_step,
                                                DurationSimulation=sim_time,
                                                SensorMap=sensor_map,
                                                Ox=Ox*amp_displacement,
                                                Oy=Oy*amp_displacement,
                                                NDelta=pml_thickness,
                                                ReflectionLimit=reflection_limit,
                                                COMPUTING_BACKEND='METAL',
                                                USE_SINGLE=True,
                                                DT=dt,
                                                QfactorCorrection=True,
                                                SelRMSorPeak=3, #we select  only RMS data
                                                SelMapsRMSPeakList=['Vx','Vy','Pressure','Sigmaxx','Sigmayy','Sigmaxy'],
                                                SelMapsSensorsList=['Vx','Vy','Pressure','Sigmaxx','Sigmayy','Sigmaxy'],
                                                SensorSubSampling=sensor_steps,
                                                DefaultGPUDeviceName=gpu_device,
                                                TypeSource=0)
    input_params_mlx = input_params_mc.copy()
    post_kernel_args_mlx = post_kernel_args_mc.copy()
    
    
    
    #%% RUN VWE CALCULATION VIA MLX
    # Kernel setup
    output_dict_mlx = vwe_mlx.kernel_setup(arguments=input_params_mlx,using_mlx=True)
    
    # Kernel execution
    results_mlx = vwe_mlx.kernel_execution(input_params_mlx,output_dict_mlx)

    # Post kernel processing
    sensor_results_mlx,last_map_results_mlx,rms_results_mlx,peak_results_mlx,InputParam_mlx = vwe_utils.VWE_post_kernel_processing(results_mlx,input_params_mlx,post_kernel_args_mlx)
    
    
    
    #%% RUN VWE CALCULATION VIA METALCOMPUTE
    # Kernel setup
    output_dict_mc = vwe_mc.kernel_setup(arguments=input_params_mc,using_mlx=False)

    # Kernel execution
    results_mc = vwe_mc.kernel_execution(input_params_mc,output_dict_mc)

    # Post kernel processing
    sensor_results_mc,last_map_results_mc,rms_results_mc,peak_results_mc,InputParam_mc = vwe_utils.VWE_post_kernel_processing(results_mc,input_params_mc,post_kernel_args_mc)
    
    # %% PLOT Function
    def plot_results(moi=None):
        if moi == "last_map":
            output_vars = ['Vx','Vy','Pressure','Sigma_xx','Sigma_yy','Sigma_xy']
        else:
            output_vars = ['Vx','Vy','Pressure','Sigmaxx','Sigmayy','Sigmaxy']
        num_vars = len(output_vars)

        fig, axes = plt.subplots(nrows=num_vars, ncols=3, figsize=(16, 6*num_vars))
        fig.suptitle(f"Output Maps for {moi} (MC vs MLX)",fontsize=18)

        # Iterate through rows
        images = []
        for index,output_key in enumerate(output_vars):
            
            # Grab results
            if moi == "last_map":
                results_mc = last_map_results_mc[output_key].T
                results_mlx = last_map_results_mlx[output_key].T
            elif moi == "rms_results":
                results_mc = rms_results_mc[output_key].T
                results_mlx = rms_results_mlx[output_key].T
            elif moi == "peak_results":
                results_mc = peak_results_mc[output_key].T
                results_mlx = peak_results_mlx[output_key].T
            elif moi == "sensor_results":
                results_mc = sensor_results_mc[output_key]
                results_mlx = sensor_results_mlx[output_key]
            else:
                raise Exception("Invalid output map given")
            
            # Plot row
            row = index
            
            if moi == 'sensor_results':
                for n, index2 in enumerate( InputParam_mc['IndexSensorMap']): 
                    i=int(index2%N1)
                    j=int(index2/N1)
                    if i==int(N1/2) and j==int(N2/2):
                        CentralPoint=n
                im1 = axes[row,0].plot(sensor_results_mc['time']*1e6,results_mc[CentralPoint])
                im2 = axes[row,1].plot(sensor_results_mlx['time']*1e6,results_mlx[CentralPoint])
                im3 = axes[row,2].plot(sensor_results_mc['time']*1e6,results_mc[CentralPoint]-results_mlx[CentralPoint])
                
            else:
                im1 = axes[row,0].imshow(results_mc)
                im2 = axes[row,1].imshow(results_mlx)
                im3 = axes[row,2].imshow(abs(results_mc-results_mlx))
            
            if moi != 'sensor_results':
                # Add colourbars
                plt.colorbar(im1, ax=axes[row,0])
                plt.colorbar(im2, ax=axes[row,1])
                plt.colorbar(im3, ax=axes[row,2])
            
            # Add titles
            axes[row,0].set_title(f'Metal Compute',fontsize='16')
            axes[row,1].set_title(f'MLX',fontsize='16')
            axes[row,2].set_title(f'Differences',fontsize='16')
            
            # Add ylabel
            axes[row,0].set_ylabel(output_key+'\n', fontsize='16')
            
            # Calculate and print difference metrics    
            dice_score = vwe_utils.calc_dice_coeff(results_mc,results_mlx)
            mse = mean_squared_error(results_mc,results_mlx)
            nrmse = normalized_root_mse(results_mc,results_mlx,normalization='min-max')
            
            print(output_key)
            print(f"DICE Score: {dice_score}")
            print(f"Mean square error: {mse}")
            print(f"Normalized root mean square error: {nrmse}\n")
            
        # Make room for the suptitle by shrinking the layout area
        plt.tight_layout(rect=[0.1,0,1,0.98])
    
    # Select map of interest    
    plot_results(moi="last_map")
    plot_results(moi="rms_results")
    plot_results(moi="peak_results")
    plot_results(moi="sensor_results")
    
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
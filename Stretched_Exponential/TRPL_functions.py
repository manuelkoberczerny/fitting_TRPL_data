"""
Routine to Estimate Carrier Lifetimes from TCSPC Decays
M. Kober-Czerny, 03-Sep-2022
"""
import os
import numpy as np
import pandas as pd
from scipy.special import gamma
from lmfit import Parameters, minimize
import ipywidgets as widgets


""" Laser Settings"""
def make_widgets(data_folder):

    # Fileselector
    all_files = os.listdir(data_folder)
    filtered_files = [k for k in all_files if '.dat' in k]
    
    file_selector = widgets.SelectMultiple(options=filtered_files, description='Files')
    
    folder = os.getcwd()
    Ref_Files = os.listdir(str(folder + '\TRPL_Files'))[::-1]
    reference_selector = widgets.Select(options=Ref_Files, description='Ref. Files')

    return file_selector, reference_selector

def unpack_Info(args):
    names, info = np.loadtxt(args, unpack=True, skiprows=1, max_rows=24, delimiter=':', dtype=str, encoding='unicode_escape')
    sample_name = str(info[np.where(names == '  Sample ')])[2:-2]
    wavelength = float(str(info[np.where(names == '  Exc_Wavelength ')])[3:-4])  # in nm
    sync_frequency = float(str(info[np.where(names == '  Sync_Frequency ')])[3:-4])  # in Hz
    signal_rate = float(str(info[np.where(names == '  Signal_Rate ')])[3:-5])  # in cps
    pile_up = signal_rate / sync_frequency * 100  # in %

    attenuation = str(info[np.where(names == '  Exc_Attenuation ')])[3:-6]
    if attenuation == 'open':
        attenuation = 1
    else:
        attenuation = float(attenuation[0:-1]) / 100

    return pile_up, attenuation, wavelength, sync_frequency, sample_name



def Fluence_Calc(wavelength, laser_reference_file, laser_intensity):
    """ Unpack Ref Data File"""
    laser_folder = os.getcwd()
    wl400, wl505, wl630 = np.loadtxt(str(laser_folder + '\TRPL_Files\\'+laser_reference_file), unpack=True, skiprows=1)

    intensity = laser_intensity

    if wavelength == 397.7:

        laser_fluence = wl400[0]*intensity+wl400[1]

    elif wavelength == 505.5:

        laser_fluence = wl505[0] * intensity + wl505[1]

    elif wavelength == 633.8:

        laser_fluence = wl630[0] * intensity + wl630[1]


    return laser_fluence



def unpack_Data(FileName):
    """ Unpack Data File"""

    ## First the data is imported, the background removed, the maximum shifted to t=0 and everything is normalized to Exc_Density

    rows_to_skip = 73
    time1 = None
    while time1 is None:
        try:
            time1, Data1 = np.loadtxt(FileName, unpack=True, skiprows= rows_to_skip, encoding='unicode_escape')
        except:
            rows_to_skip += 1
            pass
        else:
            time1, Data1 = np.loadtxt(FileName, unpack=True, skiprows = rows_to_skip, encoding='unicode_escape')
            
    Data1 = np.array(Data1)
    len_Data = len(Data1)

    """ Cut Data into Shape"""
    Data2, time2, max_locator = cut_data(Data1, time1)
    
    """ Find Background from before Pulse"""
    Data3 = remove_bckg(max_locator, Data2)
       
   
    """ Normalize Data"""
    Data4 = Data3 / Data3[np.where(time2 == 0)]
    
    Data3[Data3 <= 0] = np.nan


    return time2, Data4, max_locator, len_Data



def cut_data(Data, time):
    max_locator = np.argmax(Data)
    dtime = time[2] - time[1]
    pre_zero_time = np.linspace(max_locator, 1, max_locator)
    pre_zero_time = pre_zero_time * dtime * -1
    time = np.append(pre_zero_time, time)

    Data = np.append(Data, pre_zero_time * 0)

    return Data, time, max_locator



def remove_bckg(max_locator, Data):
    # To remove the background the median of the 30% of the highest values before the pulse are substracted
    Data_mask = np.array(Data[1:max_locator - 4])
    Bckg = np.median(Data_mask)
    Data = Data - Bckg

    return Data



def make_Dataframe(time, data_raw, len_Data, max_locator):

    Data = pd.DataFrame()

    ## The data is cut to the correct lengths and stored in a pd.Dataframe

    Data['Time'] = time[0]
    Data['0'] = data_raw[0]


    i = 1
    while i < len(len_Data):

        Data2 = data_raw[i]

        if max_locator[i] != max_locator[0]:
            if max_locator[i] > max_locator[0]:
                a = max_locator[i]-max_locator[0]
                Data2 = Data2[a:]

            elif max_locator[i] < max_locator[0]:
                Data2 = np.append(np.zeros(max_locator[0]-max_locator[i]),Data2)

        if len(Data2) != len(np.array(Data['Time'])):
            a2 = np.abs(len(np.array(Data['Time'])) - len(Data2))

            if len(Data2) > len(np.array(Data['Time'])):
                Data2 = Data2[0:-a2]
            else:
                Data2 = np.append(Data2,np.zeros(a2))

        Data[str(i)] = Data2

        i += 1


    max_locator = max_locator[0]

    return Data, max_locator



def one_over_e_Lifetime(Data, param_dict):
    
    for i, _ in enumerate(Data.columns.values[1:]):

        marker = np.where(Data[param_dict['Sample Name'][i]] < 1 / np.e)
        marker = np.array(Data['Time'])[marker]
        marker = marker[np.where(marker > 0)]
        marker = marker[0:5]
        marker1 = np.median(marker)

        param_dict.loc[i,'tau-1/e (ns)'] = np.round(np.array(marker1),1)

    return param_dict



def time_range(Fit_range, Data):
    time = np.array(Data['Time'])
    limit = np.where((time >= Fit_range[0]) & (time < Fit_range[1]))
    nan_beg = np.empty(np.min(limit))
    nan_beg[:] = np.nan
    nan_end = np.empty(len(time) - np.amax(limit))
    nan_end[:] = np.nan
    time_fit = time[limit]

    return time_fit, nan_beg, nan_end, limit



def residual2(params, time_fit, Data, limit, i):

    Data_fit = np.array(Data)[limit]

    tau = params[f'tau_{i + 1}'].value
    beta = params[f'beta_{i + 1}'].value
    A = params[f'A_{i + 1}'].value

    N_calc = A*np.exp(-(time_fit/tau)**beta)

    resid = np.sqrt(Data_fit)-np.sqrt(N_calc)

    return resid



def resid_global(params_ode, time_fit, Data, limit, resid, param_dict):

    for i,_ in enumerate(Data.columns.values[1:]):
        data = Data[param_dict['Sample Name'][i]]
        resid[i,:] = residual2(params_ode, time_fit, data, limit, i)#* time_fit

    return resid.flatten()



def post_Fitting(results_Model2, limit, time_fit, empty_dat, nan_beg, nan_end, i):
    yfit = residual2(results_Model2.params, time_fit, empty_dat, limit, i)**2

    yfit = np.append(yfit, nan_end)
    yfit = np.append(nan_beg, yfit)

    return yfit[:-1]




def import_data(data_folder, FileNames, laser_reference_file, laser_intensity):
    

    pile_up = []
    sample_name = []
    time = []
    data = []
    max_locator = []
    len_Data = []
    laser_fluence_old = []
    laser_fluence_list = []
    
    for FileName in FileNames:
        
        pile_up_1, attenuation, exc_wavelength, frequency, sample_name_1 = unpack_Info(f'{data_folder}\{FileName}')
        pile_up.append(pile_up_1)
        sample_name.append(sample_name_1)

        ## Steps to estimate Fluence

        laser_fluence = Fluence_Calc(exc_wavelength, laser_reference_file, laser_intensity)  # in cm-2
        laser_fluence = laser_fluence * attenuation # in cm-2
        laser_fluence_old.append(laser_fluence * (299792458 * 6.6261e-34) / (exc_wavelength * 1e-9) * 1e9)  # in nJ cm-2
        laser_fluence_list.append(laser_fluence)


        ### Here, the data is extracted from the file
        time1, Data3, max_locator1, len_Data1 = unpack_Data(f'{data_folder}\{FileName}')
        data.append(Data3)
        time.append(time1)
        max_locator.append(max_locator1)
        len_Data.append(len_Data1)
   

    for i, _ in enumerate(FileNames):
        len_Data[i] = len_Data[i]+len_Data[i-1]


    Data, max_locator = make_Dataframe(time, data, len_Data, max_locator)

    param_dict = pd.DataFrame(columns=['Sample Name','Fluence(cm-2)','tau_char (ns)','stretch','tau_avg (ns)','tau-1/e (ns)','pile_up_rate (%)','t=0-value','Fluence_old (nJ cm-2)'])

    param_dict.loc[:,'Sample Name'] =  sample_name
    param_dict.loc[:,'pile_up_rate (%)'] =  pile_up
    param_dict.loc[:,'Fluence(cm-2)'] = laser_fluence_list
    param_dict.loc[:,'Fluence_old (nJ cm-2)'] = laser_fluence_old


    for i, _ in enumerate(FileNames):
        Data = Data.rename({str(i): param_dict['Sample Name'][i]}, axis='columns')


    return Data, param_dict



def fit_stretched_exponential(Data, Fit_range, t_0_equals_1, share_stretch_factor, param_dict, data_folder):
    
    #### Define most important Meta-Parameters
    time_fit, nan_beg, nan_end, limit = time_range(Fit_range, Data)
    resid = np.empty(shape = (len(Data.columns.values[1:]), len(time_fit)))  #empty_dat

    ### Define Fitting Parameters
    params_ode = Parameters()
    for i, _ in enumerate(Data.columns.values[1:]):
        params_ode.add(f'tau_{i + 1}', value=600, min=0, max=100000, vary=True)
        params_ode.add(f'beta_{i + 1}', value=.5, min=0, max=1, vary=True)
        params_ode.add(f'A_{i + 1}', value=1.0, min=0.5, max=2, vary=True)

        if t_0_equals_1 == True:
            params_ode[f'A_{i + 1}'].vary = False

    if share_stretch_factor == True:
        for y, _ in enumerate(Data.columns.values[1:-1]):
            params_ode[f'beta_{y + 2}'].expr = 'beta_1'

    ### Fitting the Data
    results_Model2 = minimize(resid_global, params_ode, method='leastsq', args=(time_fit, Data, limit, resid, param_dict), nan_policy='omit')

    tau_fit = ()
    beta_fit = ()
    A_parameter = ()

    Data_fit = Data.copy()

    ### Extract important Parameters from Fit
    for i, _ in enumerate(Data.columns.values[1:]):
        yfit1 = post_Fitting(results_Model2, limit, time_fit, Data_fit[param_dict['Sample Name'][i]]*0, nan_beg, nan_end, i)
        Data_fit[str('Fit_' + param_dict['Sample Name'][i])] = yfit1
        tau_fit = np.append(tau_fit, results_Model2.params[f'tau_{i + 1}'].value)
        beta_fit = np.append(beta_fit, results_Model2.params[f'beta_{i+1}'].value)
        A_parameter = np.append(A_parameter, results_Model2.params[f'A_{i + 1}'].value)

    gamma_fit = gamma(1/beta_fit)
    tau_avg = tau_fit/beta_fit*gamma_fit

    param_dict.loc[:,'tau_char (ns)'] = np.round(tau_fit,2)
    param_dict.loc[:,'stretch'] = np.round(beta_fit,2)
    param_dict.loc[:,'tau_avg (ns)'] = np.round(tau_avg, 2)
    param_dict.loc[:,'A-parameter'] = np.round(A_parameter, 2)
    
    Data_save = Data_fit.replace(np.nan, str('NaN'), regex=True)
    
    file_name = param_dict['Sample Name'][0]
    Data_save.to_csv(f'{data_folder}/{file_name}_processed_stretch.txt', sep='\t', mode='w')
    param_dict.to_csv(f'{data_folder}/{file_name}_fit-values_stretch.txt', sep='\t', index= False, mode='w')

    return Data_fit, param_dict


#def plot_stretched_exponential_fit(Data, param_dict):
#    #### Define Plots
#color1 = iter(cm.plasma(np.linspace(0, 1, (num_rows+1))))
#
#fig = plt.figure(figsize=(16, 9))
#fig.suptitle('Transient Photoluminescence (TCSPC) Fitting - ' + str(date.today().strftime("%d-%b-%Y")), fontsize=20)
#plt.subplots_adjust(wspace=0.28, hspace=0.35)
#gs = gridspec.GridSpec(2, 3)
#
#
#ax1 = fig.add_subplot(gs[:2, :2])
#ax1.set_title('Photoluminescence Transients (Lin-Log)', fontsize=14)
#ax1.set_xlim(-40, Fit_range[1] + 100)
#ax1.set_ylim(1e-6, 1.5)
#ax1.set_xlabel('Time after Pulse [ns]')
#ax1.set_ylabel('Normalized PL [a.u.]')
#
#
#ax2 = fig.add_subplot(gs[0, 2])
#ax2.set_title('Estimated Lifetimes', fontsize = 14)
#ax2.set_ylabel('Lifetime [ns]')
#ax2.set_xlabel('Fluence [cm⁻²]')
#
#
#ax3 = fig.add_subplot(gs[1,2])
#ax3.set_title('Other Parameters', fontsize = 14)
#ax3.set_ylim(0,2)
#ax3.set_ylabel('Stretch Factor or A-Parameter [-]')
#ax3.set_xlabel('Fluence [cm⁻²]')
#
#
#### Fill plots with content
#for i in range(num_rows):
#    color2 = next(color1)
#
#    ax1.semilogy(Data['Time'], Data[str(i)], 'o', alpha=(0.25), c=color2,label=str(sample_name[i] + " (" + "{:.2f}".format(laser_fluence_old[i]) + "  nJ cm⁻²)"))
#for i in range(num_rows):
#    ax1.semilogy(Data['Time'], Data[str('Fit_' + str(i))], zorder=100, c='mediumseagreen', linewidth=3)
#
#
#ax1.vlines(x=[Fit_range[0], Fit_range[1]], ymin=1e-6, ymax=1.5, color='grey', linestyles='--')
#ax1.annotate(str("Fit from " + "{:.0f}".format(Fit_range[0]) + "  to  " + "{:.0f}".format(Fit_range[1])+ " ns"),xy=(0.04,0.12),xycoords='axes fraction',c='black', fontsize = 10)
#
#pile_up_max = np.amax(pile_up)
#if pile_up_max < 5.0:
#    ax1.annotate(str("Pile-up Rate ≤ " + "{:.1f}".format(pile_up_max) + "%"),xy=(0.04,0.09),xycoords='axes fraction', c='green', fontsize=10)
#elif (pile_up_max > 5.0) & (pile_up_max < 8.0):
#    ax1.annotate(str("Pile-up Rate ≤ " + "{:.1f}".format(pile_up_max) + "%  (should be mentioned)"),xy=(0.04,0.09),xycoords='axes fraction', c='orange', fontsize=10)
#else:
#    ax1.annotate(str("Pile-up Rate ≤ " + "{:.1f}".format(pile_up_max) + "%  (data probably affected)"),xy=(0.04,0.09),xycoords='axes fraction', c='red', fontsize=10)
#
#ax1.annotate(str("R² value: " +  "{:.3f}".format(np.mean(R2_val)) + "  +/-  " + "{:.3f}".format(np.std(R2_val))),xy=(0.04,0.06),xycoords='axes fraction',c='black', fontsize=10)
#ax1.annotate(str("Wavelength: " +  "{:.0f}".format(exc_wavelength) + " nm"),xy=(0.04,0.03),xycoords='axes fraction',c='black', fontsize=10)
#
#
#ax1.legend()
#
#
#
#ax2.semilogx(laser_fluence_list, e_Lifetime, marker='o', c='black', label="1/e Lifetime Data")
#ax2.errorbar(laser_fluence_list, e_Lifetime, yerr=(err1, err2), c='black')
#
#ax2.semilogx(laser_fluence_list, tau_fit, marker='o', c='mediumseagreen', label="Characteristic Lifetime")
#ax2.semilogx(laser_fluence_list, tau_avg, marker='o', c=color2, label="Average Lifetime")
#ax2.legend()
#
#
#ax3.semilogx(laser_fluence_list, beta_fit, marker='o', c='darkblue',label='Stretch Factor')
#ax3.semilogx(laser_fluence_list, A_parameter, marker='o', c='lightblue',label='A-parameter')
#
#ax3.legend()
#
#
#
#
## Store Data and Parameters
#
#Fit_Values = pd.DataFrame(index=[sample_name],columns=['Fluence (nJ cm-2)','tau_char (ns)','stretch','tau_avg (ns)','tau-1/e (ns)','pile-up (%)','A-parameter','R_sqr'])
#
#for i in range(num_rows):
#    Fit_Values.loc[sample_name[i],'Fluence (nJ cm-2)'] = round(laser_fluence_list[i],2)
#    Fit_Values.loc[sample_name[i], 'tau_char (ns)'] = round(tau_fit[i],2)
#    Fit_Values.loc[sample_name[i], 'stretch'] = round(beta_fit[i],2)
#    Fit_Values.loc[sample_name[i], 'tau_avg (ns)'] = round(tau_avg[i], 2)
#    Fit_Values.loc[sample_name[i], 'tau-1/e (ns)'] = round(e_Lifetime[i],1)
#    Fit_Values.loc[sample_name[i], 'pile-up (%)'] = round(pile_up[i],1)
#    Fit_Values.loc[sample_name[i], 'A-parameter'] = round(A_parameter[i], 2)
#    Fit_Values.loc[sample_name[i], 'R_sqr'] = round(R2_val[i],3)
#
#    Data = Data.rename({str(i): sample_name[i]}, axis='columns')
#    Data = Data.rename({str('Fit_' + str(i)): str('Fit_' + sample_name[i])}, axis='columns')
#
#Data = Data.replace(np.nan, str('NaN'), regex=True)
#
#
#print('Fitting Results:')
#print(Fit_Values.to_string())
#
#
#
#
#
#
#
#
#### Saving the Data and Figure
#Fit_Values.to_csv(str(PurePath(args.directory).joinpath(str(args.short_name).replace('.dat', '_fit-values_stretch.txt'))), sep='\t', index= True, mode='w')
#Data.to_csv(str(PurePath(args.directory).joinpath(str(args.short_name).replace('.dat', '_processed_stretch.txt'))), sep='\t', mode='w')
#plt.savefig(str(PurePath(args.directory).joinpath(str(args.short_name).replace('.dat', '_figure_stretch.png'))), format='png',dpi=100)
#
#
#plt.show()
#
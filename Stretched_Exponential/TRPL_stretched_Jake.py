"""
Routine to Estimate Carrier Lifetimes from TCSPC Decays
M. Kober-Czerny, 03-Sep-2022
"""
import os
from pathlib import Path, PurePath
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from gooey import Gooey, GooeyParser
import numpy as np
import pandas as pd
from scipy.special import gamma
from lmfit import Parameters, minimize
from scipy.signal import medfilt



""" Laser Settings"""

folder = os.getcwd()
Ref_Files = os.listdir(str(folder + '\TRPL_Files'))

# Use flag --ignore-gooey if you want to use the command line
@Gooey(advanced=True,          # toggle whether to show advanced config or not
       default_size=(800, 700),   # starting size of the GUI
       show_success_modal = False,
       navigation = "Tabbed",
       sidebar_groups=True,
       return_to_config = True,
       sidebar_title = "Navigation",
       image_dir=str(folder + '\Icon\\')
)


def get_args():
    """Get arguments and options"""
    parser = GooeyParser(description='Fitting TCSPC Transients to Obtain Carrier Lifetimes')

    req = parser.add_argument_group('Main', gooey_options={'columns': 1})
    req.add_argument('-dp', '--data_path', nargs='*', widget="MultiFileChooser", help="Path to the Datafile(s)", gooey_options={'wildcard':"TRPL' files (TRPL_*.dat) |TRPL_*.dat|" "All files (*.*)|*.*",'full_width':True})

    opt = parser.add_argument_group('Measurement Settings', gooey_options={'columns': 2})
    opt.add_argument('-int', '--laser_intensity', default="9", type=float, help="Laser Intensity (from settings)")
    opt.add_argument('-ref', '--laser_reference_file', widget="Dropdown",default = Ref_Files[-1], choices=Ref_Files,
                      type=str,
                      help="Pick the Laser Reference File")

    opt3 = parser.add_argument_group('Fitting Parameters', gooey_options={'columns': 2})
    opt3.add_argument('-fstart', '--fit_start', default=0, type=float, help="Fit start [ns]")
    opt3.add_argument('-fend', '--fit_end',  default=1500, type=float, help="Fit until... [ns]")
    opt3.add_argument('-sl', '--share_stretch', action='store_true', help="Same Stretch-Factor for all Measurements")
    opt3.add_argument('-s2', '--fix_A_paremeter', action='store_true', help="fix A-parameter to 1.0")

    args = parser.parse_args()
    args.directory = Path(args.data_path[0]).resolve().parent
    args.short_name = Path(args.data_path[0]).name
    args.cwd = Path.cwd()
    return args



def unpack_Info(args):

    names, info = np.loadtxt(args, unpack=True, skiprows=1, max_rows=36, delimiter=' : ', dtype=str)
    sample_name = str(info[np.where(names == '  Sample')])[2:-2]

    exc_wavelength = float(str(info[np.where(names == '  Exc_Wavelength')])[2:-4])      # in nm

    det_wavelength = float(str(info[np.where(names == '  Det_Wavelength')])[2:-4])      # in nm

    sync_frequency = float(str(info[np.where(names == '  Sync_Frequency')])[2:-4])  # in Hz
    signal_rate = float(str(info[np.where(names == '  Signal_Rate')])[2:-5])        # in cps
    pile_up = signal_rate/sync_frequency*100                                        # in %

    attenuation = str(info[np.where(names == '  Exc_Attenuation')])[2:-6]
    if attenuation == 'open':
        attenuation = 1
    else:
        attenuation = float(attenuation[0:-1])/100

    return pile_up, attenuation, exc_wavelength, sync_frequency, sample_name, det_wavelength



def Fluence_Calc(wavelength, args):
    """ Unpack Ref Data File"""
    wl400, wl505, wl630 = np.loadtxt(str(folder + '\TRPL_Files\\'+args.laser_reference_file), unpack=True, skiprows=1)

    intensity = args.laser_intensity

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

    time1, Data1 = np.loadtxt(FileName, unpack=True, skiprows= 76)
    Data1 = np.array(Data1)
    len_Data = len(Data1)

    """ Cut Data into Shape"""
    Data2, max_locator,time2 = cut_data(Data1, time1)

    """ Remove Background from before Pulse"""

    Data3 = remove_bckg(Data2, max_locator)

    """ Normalize Data"""

    Data4 = Data3 / Data3[np.argmax(Data3)]
    Data4[Data4 < 0] = np.nan


    return time2, Data4, max_locator, len_Data



def cut_data(Data, time):
    max_locator = np.argmax(Data)
    dtime = time[2] - time[1]
    pre_zero_time = np.linspace(max_locator, 1, max_locator)
    pre_zero_time = pre_zero_time * dtime * -1
    time = np.append(pre_zero_time, time)

    Data = np.append(Data, pre_zero_time * 0)

    return Data, max_locator, time



def remove_bckg(Data, max_locator):
    # To remove the background the median of the 30% of the highest values before the pulse are substracted
    Data_mask = np.array(Data[1:max_locator - 4])
    Bckg = np.median(Data_mask)
    Data = Data - Bckg

    return Data



def make_Dataframe(time, data_raw, len_Data, max_locator):

    Data = pd.DataFrame()
    Data_raw = pd.DataFrame()

    ## The data is cut to the correct lengths and stored in a pd.Dataframe

    Data['Time'] = time[0]
    Data_raw['0'] = data_raw[0]


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

        Data_raw[str(i)] = Data2

        i += 1

    Data['0'] = Data_raw.mean(axis=1)
    max_locator = max_locator[0]

    return Data, max_locator



def one_over_e_Lifetime(Data):
    marker = np.where(Data['0'] < 1 / np.e)
    marker = np.array(Data['Time'])[marker]
    marker = marker[np.where(marker > 0)]
    marker = marker[0:5]
    marker1 = np.median(marker)
    marker_err1 = abs(np.quantile(marker, 0.25) - marker1)
    marker_err2 = abs(np.quantile(marker, 0.75) - marker1)


    return marker1, marker_err1, marker_err2



def time_range(Fit_range, Data):
    time = np.array(Data['Time'])
    limit = np.where((time >= Fit_range[0]) & (time < Fit_range[1]))
    nan_beg = np.empty(np.min(limit))
    nan_beg[:] = np.nan
    nan_end = np.empty(len(time) - np.amax(limit))
    nan_end[:] = np.nan
    time_fit = time[limit]

    return time_fit, nan_beg, nan_end, limit



def residual2(params, time_fit, Data, limit,  i):

    Data_fit = np.array(Data['0'])[limit]

    tau = params[f'tau'].value
    beta = params[f'beta'].value
    A = params[f'A'].value

    dndt = A*np.exp(-(time_fit/tau)**beta)

    resid = Data_fit-dndt

    return resid



def resid_global(params_ode, time_fit, photo_cond, limit, resid, i):

    resid = np.abs(residual2(params_ode, time_fit, photo_cond, limit, i))* time_fit

    return resid



def R2_val_calculation(Data,limit,yfit):

    Data_fit_test = np.array(Data)[limit]
    residual = (Data_fit_test - yfit)
    residual = np.nan_to_num(residual, nan=0)

    y2fit = np.nan_to_num(yfit, nan=0)
    R2_val_calc = 1 - np.sum(residual ** 2) / np.sum((y2fit - np.mean(y2fit)) ** 2)

    return R2_val_calc



def post_Fitting(results_Model2, Data, limit, time_fit, empty_dat, i, nan_beg, nan_end):
    yfit = -residual2(results_Model2.params, time_fit, empty_dat, limit, i)

    R2_val_calc = R2_val_calculation(Data, limit, yfit)

    yfit = np.append(yfit, nan_end)
    yfit = np.append(nan_beg, yfit)

    return R2_val_calc, yfit[:-1]











############################
""" MAIN PART OF THE CODE"""
############################

### This part of the code gets the filename(s) and makes a list out of it for later use


args = get_args()

FileName = args.data_path
global num_rows
num_rows = len(FileName)

pile_up = []
sample_name = []

time = []
data = []
max_locator = []
len_Data = []



### Here, the additional information included in the Data header is imported and used to calculate the first important numbers

for i in range(num_rows):
    pile_up_1, attenuation, exc_wavelength, frequency, sample_name_1, det_wavelength = unpack_Info(FileName[i])
    pile_up.append(pile_up_1)
    sample_name.append(sample_name_1)

    ## Steps to estimate Fluence

    laser_fluence = Fluence_Calc(exc_wavelength, args)  # in cm-2
    laser_fluence = laser_fluence * attenuation # in cm-2
    laser_fluence_old = laser_fluence * (299792458 * 6.6261e-34) / (exc_wavelength * 1e-9) * 1e9  # in nJ cm-2


    ### Here, the data is extracted from the file
    time1, Data3, max_locator1, len_Data1 = unpack_Data(FileName[i])
    data.append(Data3)
    time.append(time1)
    max_locator.append(max_locator1)
    len_Data.append(len_Data1)


### Length of Data is changed so it fits into a single Dataframe

i = 1
while i < num_rows:
    len_Data[i] = len_Data[i]+len_Data[i-1]
    i += 1


Data, max_locator = make_Dataframe(time, data, len_Data, max_locator)



""" FITTING STARTS HERE"""
### 1/e Lifetime Calculation
i = 0

e_Lifetime1, err1a, err2a = one_over_e_Lifetime(Data)
e_Lifetime = e_Lifetime1
err1 = err1a
err2 = err2a



### Define most important Fitting Parameters
Fit_range = np.array([args.fit_start, args.fit_end])



time_fit, nan_beg, nan_end, limit = time_range(Fit_range, Data)


resid = np.empty(shape = len(time_fit))
empty_dat = Data*0
empty_dat[empty_dat == np.nan] = 0

params_ode = Parameters()
params_ode.add(f'tau', value=600, min=0.1, max=100000, vary=True)
params_ode.add(f'beta', value=.5, min=0.1, max=1, vary=True)
params_ode.add(f'A', value=1.0, min=0.5, max=2, vary=True)

if args.fix_A_paremeter == True:
    params_ode[f'A'].vary = False



### Fitting the Data
results_Model2 = minimize(resid_global, params_ode, method='leastsq', args=(time_fit, Data, limit, resid, 0), nan_policy='omit')




### Extract important Parameters from Fit
R2_val_calc, yfit1 = post_Fitting(results_Model2, Data['0'], limit, time_fit, empty_dat, i, nan_beg, nan_end)
R2_val = R2_val_calc
Data[str('Fit_' + str(i))] = yfit1
tau_fit = results_Model2.params[f'tau'].value
beta_fit = results_Model2.params[f'beta'].value
A_parameter = results_Model2.params[f'A'].value

gamma_fit = gamma(1/np.array(beta_fit))
tau_avg = np.array(tau_fit)/np.array(beta_fit)*np.array(gamma_fit)


""" PLOTTING OF DATA """
### Define Plots
color1 = iter(cm.plasma(np.linspace(0, 1, (num_rows+1))))

fig = plt.figure(figsize=(16, 9))
fig.suptitle('Transient Photoluminescence (TCSPC) Fitting - ' + str(date.today().strftime("%d-%b-%Y")), fontsize=20)
plt.subplots_adjust(wspace=0.28, hspace=0.35)
gs = gridspec.GridSpec(2, 3)


ax1 = fig.add_subplot(gs[:2, :2])
ax1.set_title('Photoluminescence Transients (Lin-Log)', fontsize=14)
ax1.set_xlim(-40, Fit_range[1] + 100)
ax1.set_ylim(1e-6, 1.5)
ax1.set_xlabel('Time after Pulse [ns]')
ax1.set_ylabel('Normalized PL [a.u.]')


ax2 = fig.add_subplot(gs[0, 2])
ax2.set_title('Estimated Lifetimes', fontsize = 14)
ax2.set_ylabel('Lifetime [ns]')
ax2.set_xlabel('Fluence [nJ cm⁻²]')


ax3 = fig.add_subplot(gs[1,2])
ax3.set_title('Other Parameters', fontsize = 14)
ax3.set_ylim(0,2)
ax3.set_ylabel('Stretch Factor or A-Parameter [-]')
ax3.set_xlabel('Fluence [nJ cm⁻²]')


### Fill plots with content
color2 = next(color1)

ax1.semilogy(Data['Time'], Data['0'], 'o', alpha=(0.25), c=color2,label=str(sample_name[i] + " (" + "{:.2f}".format(laser_fluence_old) + "  nJ cm⁻²)"))
ax1.semilogy(Data['Time'], Data[str('Fit_' + str(i))], zorder=100, c='mediumseagreen', linewidth=3)


ax1.vlines(x=[Fit_range[0], Fit_range[1]], ymin=1e-6, ymax=1.5, color='grey', linestyles='--')
ax1.annotate(str("Fit from " + "{:.0f}".format(Fit_range[0]) + "  to  " + "{:.0f}".format(Fit_range[1])+ " ns"),xy=(0.04,0.12),xycoords='axes fraction',c='black', fontsize = 10)

pile_up_max = np.amax(pile_up)
if pile_up_max < 5.0:
    ax1.annotate(str("Pile-up Rate ≤ " + "{:.1f}".format(pile_up_max) + "%"),xy=(0.04,0.09),xycoords='axes fraction', c='green', fontsize=10)
elif (pile_up_max > 5.0) & (pile_up_max < 8.0):
    ax1.annotate(str("Pile-up Rate ≤ " + "{:.1f}".format(pile_up_max) + "%  (should be mentioned)"),xy=(0.04,0.09),xycoords='axes fraction', c='orange', fontsize=10)
else:
    ax1.annotate(str("Pile-up Rate ≤ " + "{:.1f}".format(pile_up_max) + "%  (data probably affected)"),xy=(0.04,0.09),xycoords='axes fraction', c='red', fontsize=10)

ax1.annotate(str("R² value: " +  "{:.3f}".format(np.mean(R2_val)) + "  +/-  " + "{:.3f}".format(np.std(R2_val))),xy=(0.04,0.06),xycoords='axes fraction',c='black', fontsize=10)
ax1.annotate(str("Wavelength: " +  "{:.0f}".format(exc_wavelength) + " nm"),xy=(0.04,0.03),xycoords='axes fraction',c='black', fontsize=10)


ax1.legend()



ax2.semilogx(laser_fluence_old, e_Lifetime, marker='o', c='black', label="1/e Lifetime Data")
#ax2.errorbar(laser_fluence_old, e_Lifetime, yerr=(err1, err2), c='black')

ax2.semilogx(laser_fluence_old, tau_fit, marker='o', c='mediumseagreen', label="Characteristic Lifetime")
ax2.semilogx(laser_fluence_old, tau_avg, marker='o', c=color2, label="Average Lifetime")
ax2.legend()


ax3.semilogx(laser_fluence_old, beta_fit, marker='o', c='darkblue',label='Stretch Factor')
ax3.semilogx(laser_fluence_old, A_parameter, marker='o', c='lightblue',label='A-parameter')

ax3.legend()




# Store Data and Parameters

Fit_Values = pd.DataFrame(index=[sample_name],columns=['Fluence (nJ cm-2)','tau_char (ns)','stretch','tau_avg (ns)','tau-1/e (ns)','pile-up (%)','A-parameter','R_sqr'])

Fit_Values.loc[sample_name[i],'Fluence (nJ cm-2)'] = round(laser_fluence_old,2)
Fit_Values.loc[sample_name[i], 'tau_char (ns)'] = round(tau_fit,2)
Fit_Values.loc[sample_name[i], 'stretch'] = round(beta_fit,2)
Fit_Values.loc[sample_name[i], 'tau_avg (ns)'] = round(tau_avg, 2)
Fit_Values.loc[sample_name[i], 'tau-1/e (ns)'] = round(e_Lifetime,1)
Fit_Values.loc[sample_name[i], 'pile-up (%)'] = round(np.max(pile_up),1)
Fit_Values.loc[sample_name[i], 'A-parameter'] = round(A_parameter, 2)
Fit_Values.loc[sample_name[i], 'R_sqr'] = round(R2_val,3)

#Data = Data.rename({str(i): sample_name[i]}, axis='columns')
#Data = Data.rename({str('Fit_' + str(i)): str('Fit_' + sample_name)}, axis='columns')

Data = Data.replace(np.nan, str('NaN'), regex=True)


print('Fitting Results:')
print(Fit_Values.to_string())








### Saving the Data and Figure
Fit_Values.to_csv(str(PurePath(args.directory).joinpath(str(args.short_name).replace('.dat', '_fit-values_stretch.txt'))), sep='\t', index= True, mode='w')
Data.to_csv(str(PurePath(args.directory).joinpath(str(args.short_name).replace('.dat', '_processed_stretch.txt'))), sep='\t', mode='w')
plt.savefig(str(PurePath(args.directory).joinpath(str(args.short_name).replace('.dat', '_figure_stretch.png'))), format='png',dpi=100)


plt.show()

exit()

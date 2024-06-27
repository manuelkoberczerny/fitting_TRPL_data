"""
Routine to Estimate Recombination from TCSPC Decays
M. Kober-Czerny
"""

import os
from pathlib import Path, PurePath
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from gooey import Gooey, GooeyParser
from lmfit import Parameters, minimize
from scipy.integrate import odeint
from scipy.signal import medfilt

""" Laser Settings"""

folder = os.getcwd()
Ref_Files = os.listdir(str(folder + '\TRPL_Files'))


# Use flag --ignore-gooey if you want to use the command line
@Gooey(advanced=True,  # toggle whether to show advanced config or not
       default_size=(800, 700),  # starting size of the GUI
       show_success_modal=False,
       navigation="Tabbed",
       sidebar_groups=True,
       return_to_config=True,
       sidebar_title="Navigation",
       image_dir=str(folder + '\Icon\\')
       )
def get_args():
    """Get arguments and options"""
    parser = GooeyParser(description='Fitting TCSPC Transients to Obtain Recombination Constants')

    req = parser.add_argument_group('Main', gooey_options={'columns': 1})
    req.add_argument('-dp', '--data_path', nargs='*', widget="MultiFileChooser", help="Path to the Datafile(s)",
                     gooey_options={'wildcard': "TRPL' files (TRPL_*.dat) |TRPL_*.dat|" "All files (*.*)|*.*",
                                    'full_width': True})

    opt = parser.add_argument_group('Measurement Settings', gooey_options={'columns': 2})
    opt.add_argument('-int', '--laser_intensity', default="9", type=float, help="Laser Intensity (from settings)")
    opt.add_argument('-s_abs', '--sample_absorbtance', default="0.75", type=float,
                     help="Sample Absorbtance at λ \nReflectance from 1 mm glass: \n405 nm = 0.14; 505 nm = 0.14; 633 nm = 0.06")
    opt.add_argument('-th', '--Thickness', default=500, type=float, help="Film Thickness [nm]")
    opt.add_argument('-ref', '--laser_reference_file', widget="Dropdown", default=Ref_Files[-1], choices=Ref_Files,
                     type=str,
                     help="Pick the Laser Reference File")

    opt3 = parser.add_argument_group('Fitting Parameters', gooey_options={'columns': 3},
                                     description='The program fits all selected datafiles simultaneouzly and generates one report. The fitting is not necessarily gobal.')
    opt3.add_argument('-k1', '--k1', default="1e6", type=float, help=r"SRH recombination constant [s⁻¹]")
    opt3.add_argument('-k1fix', '--k1_fixed', action='store_true', help="fixed?")
    opt3.add_argument('-k1share', '--k1_shared', action='store_true', help="k1 is shared globally")
    opt3.add_argument('-k2', '--k2', default="1e-10", type=float, help="radiative recombination constant [cm³s⁻¹]")
    opt3.add_argument('-k2fix', '--k2_fixed', action='store_true', help="fixed?")
    opt3.add_argument('-k2share', '--k2_shared', action='store_true', help="k2 is shared globally")
    opt3.add_argument('-fstart', '--fit_start', default=0, type=float, help="Fit start at ... [ns]")
    opt3.add_argument('-fend', '--fit_end', default=1500, type=float, help="Fit until ... [ns]")
    opt3.add_argument('-aed', '--additional_early_decay', action='store_true',
                      help="Additional early decay observed?")

    opt4 = parser.add_argument_group('Special Fitting Parameters', gooey_options={'columns': 2},
                                     description='These parameters are for advanced fittings and are usually not needed')
    opt4.add_argument('-odextr', '--OD_extra', default=0, type=float,
                      help="Additional OD (for cryostat, thicker samples, etc.)")
    opt4.add_argument('-x', '--x-Parameter', widget="Dropdown", type=int, default=2, choices=[1, 2],
                      help="n^x ~ PL")

    args = parser.parse_args()
    args.directory = Path(args.data_path[0]).resolve().parent
    args.short_name = Path(args.data_path[0]).name
    args.cwd = Path.cwd()
    return args


def unpack_Info(args):
    names, info = np.loadtxt(args, unpack=True, skiprows=1, max_rows=24, delimiter=' : ', dtype=str)
    sample_name = str(info[np.where(names == '  Sample')])[2:-2]

    wavelength = float(str(info[np.where(names == '  Exc_Wavelength')])[2:-4])  # in nm

    sync_frequency = float(str(info[np.where(names == '  Sync_Frequency')])[2:-4])  # in Hz
    signal_rate = float(str(info[np.where(names == '  Signal_Rate')])[2:-5])  # in cps
    pile_up = signal_rate / sync_frequency * 100  # in %

    attenuation = str(info[np.where(names == '  Exc_Attenuation')])[2:-6]
    if attenuation == 'open':
        attenuation = 1
    else:
        attenuation = float(attenuation[0:-1]) / 100

    return pile_up, attenuation, wavelength, sync_frequency, sample_name


def Fluence_Calc(wavelength, args):
    """ Unpack Ref Data File"""
    wl400, wl505, wl630 = np.loadtxt(str(folder + '\TRPL_Files\\' + args.laser_reference_file), unpack=True, skiprows=1)

    intensity = args.laser_intensity

    if wavelength == 397.7:

        laser_fluence = wl400[0] * intensity + wl400[1]

    elif wavelength == 505.5:

        laser_fluence = wl505[0] * intensity + wl505[1]

    elif wavelength == 633.8:

        laser_fluence = wl630[0] * intensity + wl630[1]

    return laser_fluence


def unpack_Data(FileName):
    """ Unpack Data File"""

    ## First the data is imported, the background removed, the maximum shifted to t=0 and everything is normalized to Exc_Density

    time1, Data1 = np.loadtxt(FileName, unpack=True, skiprows=76)
    Data1 = np.array(Data1)
    len_Data = len(Data1)

    """ Cut Data into Shape"""
    Data2, max_locator, time2 = cut_data(Data1, time1)

    """ Remove Background from before Pulse"""

    Data3 = remove_bckg(Data2, max_locator)

    """ Normalize Data"""
    data_max = Data3[max_locator]
    #Data3 = medfilt(Data3, 7)
    Data5 = Data3 / np.max(Data3)
    Data5[Data5 < 0] = np.nan

    return time2, Data5, max_locator, len_Data, data_max


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
    Data_mask = np.array(Data[1:max_locator - 3])
    Bckg = np.nanmedian(Data_mask)
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
                a = max_locator[i] - max_locator[0]
                Data2 = Data2[a:]

            elif max_locator[i] < max_locator[0]:
                Data2 = np.append(np.zeros(max_locator[0] - max_locator[i]), Data2)

        if len(Data2) != len(np.array(Data['Time'])):
            if len(Data2) > len(np.array(Data['Time'])):
                a = np.abs(len(Data2)-len(np.array(Data['Time'])))
                Data2 = Data2[0:-a]
            else:
                a = np.abs(len(Data2) - len(np.array(Data['Time'])))
                Data2 = np.append(Data2, np.zeros(a))

        Data[str(i)] = Data2

        i += 1

    max_locator = max_locator[0]

    return Data, max_locator


def one_over_e_Lifetime(Data):
    marker = np.where(Data[str(i)] < 1 / np.e)
    marker = np.array(Data['Time'])[marker]
    marker = marker[np.where(marker > 0)]
    marker = marker[0:5]
    marker1 = np.median(marker)
    marker_err1 = 0#abs(np.quantile(marker, 0.25) - marker1)
    marker_err2 = 0#abs(np.quantile(marker, 0.75) - marker1)

    return marker1, marker_err1, marker_err2


def time_range(Fit_range, Data):
    time = np.array(Data['Time'])
    limit = np.where((time > -10) & (time < Fit_range[1]))

    time_fit = time[limit]

    limit_i = Fit_range[0]
    limit_inner = np.where((time_fit >= limit_i))[0][0]

    nan_beg = np.empty(np.min(limit))
    nan_beg[:] = np.nan
    nan_end = np.empty(len(time) - np.amax(limit))
    nan_end[:] = np.nan


    return time_fit, nan_beg, nan_end, limit, limit_inner


def df_sigma(n0, time, params, i):
    def gaussian(t, mu, sig):
        return 1 / (sig * np.sqrt(2 * np.pi)) * np.exp(-(1 / 2) * ((t - mu) / sig) ** 2)

    k1 = params[f'k1_{i + 1}'].value
    k2 = params[f'k2_{i + 1}'].value
    k3 = params[f'k3_{i + 1}'].value

    sig = pulse_fwhm / (2 * np.sqrt(2 * np.log(2)))

    mu = 0
    ne = n0

    gauss_dat = gaussian(time, mu, sig)

    dnedt = Exc_Density[i] * gauss_dat - k1 * ne - k2 * ne ** 2 - k3 * ne ** 2
    return dnedt


def residual2(params_ode, time_fit, Counts, limit, limit2a, i):
    I0 = 0

    Data_fit = np.array(Counts[str(i)])[limit]
    A_ct = params_ode[f'A_ct_{i + 1}'].value
    tau_ct = params_ode[f'tau_ct_{i + 1}'].value

    dnedt = odeint(df_sigma, I0, time_fit*1e-9, args=(params_ode, i,))
    dndt = dnedt.reshape(-1)

    
    CT = np.where(time_fit > 0, (np.exp(-time_fit / tau_ct)), 0)
    dndt = ((dndt / Exc_Density[i]))**x_Parameter * (1 - A_ct) + (A_ct * CT)

    dndt = dndt/np.nanmax(dndt)
    resid = (Data_fit - dndt)

    return resid[limit2a+1:]


def resid_global(params_ode, time_fit, photo_cond, limit, limit2a, resid, i):
    while i < num_rows:
        resid[i, :] = np.abs(residual2(params_ode, time_fit, photo_cond, limit, limit2a, i)) * (time_fit[limit2a:-1]*1e9)
        i += 1

    return resid.flatten()


def post_Fitting(results_Model2, Data, limit, limit_inner, time_fit, empty_dat, i, nan_beg, nan_end):
    yfit = -residual2(results_Model2.params, time_fit, empty_dat, limit, 0, i)

    R2_val_calc = R2_val_calculation(Data, limit_inner, yfit[limit_inner:])

    yfit = np.append(yfit, nan_end)
    yfit = np.append(nan_beg, yfit)

    return R2_val_calc, yfit


def R2_val_calculation(Data,limit,yfit):

    Data_fit_test = np.array(Data)[limit]
    residual = (Data_fit_test - yfit)
    residual = np.nan_to_num(residual, nan=0)

    y2fit = np.nan_to_num(yfit, nan=0)
    R2_val_calc = 1 - np.sum(residual ** 2) / np.sum((y2fit - np.mean(y2fit)) ** 2)

    return R2_val_calc





""" Main Part of Code"""
### This part of the code gets the filename(s) and makes a list out of it for later use

args = get_args()

FileName = args.data_path
global num_rows
num_rows = len(FileName)

global Exc_Density
Exc_Density = []

global x_Parameter
x_Parameter = args.x_Parameter

global pulse_fwhm  # ~ 372 ps
pulse_fwhm = 1e-9


pile_up = []
sample_name = []

time = []
data = []
max_locator = []
data_max = []
len_Data = []
laser_fluence_old = []

### Here, the additional information included in the Data header is imported and used to calculate the first important numbers

for i in range(num_rows):
    pile_up_1, attenuation, wavelength, frequency, sample_name_1 = unpack_Info(FileName[i])
    pile_up.append(pile_up_1)
    sample_name.append(sample_name_1)

    ## Steps to calculate Laser Fluence and Excitation Density
    laser_fluence = Fluence_Calc(wavelength, args)  # in cm-2
    laser_fluence = laser_fluence * attenuation * 10 ** (- args.OD_extra)  # in cm-2
    laser_fluence_old.append(laser_fluence * (299792458 * 6.6261e-34) / (wavelength * 1e-9) * 1e9)  # in nJ cm-2
    Exc_Density.append(laser_fluence * args.sample_absorbtance / (args.Thickness * 1e-7))  # in cm-3



    ### Here, the data is combined into a dataframe
    time1, Data3, max_locator1, len_Data1, data_max1 = unpack_Data(FileName[i])
    data.append(Data3)
    time.append(time1)
    data_max.append(data_max1)
    max_locator.append(max_locator1)
    len_Data.append(len_Data1)


### Length of Data is changed so it fits into a single Dataframe
i = 1
while i < num_rows:
    len_Data[i] = len_Data[i] + len_Data[i - 1]
    i += 1

### The Dataframe is created
Data, max_locator = make_Dataframe(time, data, len_Data, max_locator)

""" FITTING STARTS HERE"""
### 1/e Lifetime Calculation

e_Lifetime = []
err1 = []
err2 = []

i = 0
while i < num_rows:
    e_Lifetime1, err1a, err2a = one_over_e_Lifetime(Data)
    e_Lifetime.append(e_Lifetime1)
    err1.append(err1a)
    err2.append(err2a)

    i += 1

### Define most important Fitting Parameters
Fit_range = np.array([args.fit_start, args.fit_end])

time_fit, nan_beg, nan_end, limit, limit_inner = time_range(Fit_range, Data)

resid = np.empty(shape = (num_rows, len(time_fit)-limit_inner-1))
empty_dat = Data * 0
empty_dat[empty_dat == np.nan] = 0

params_ode = Parameters()
for i in range(num_rows):
    params_ode.add(f'k1_{i + 1}', value=args.k1, min=0, max=1e10, vary=True)
    params_ode.add(f'k2_{i + 1}', value=args.k2, min=1e-13, max=1e-8, vary=True)
    params_ode.add(f'k3_{i + 1}', value=1e-28, max=1e-27, vary=False)
    params_ode.add(f'A_ct_{i + 1}', value=0, min=0, max=1.000, vary=False)
    params_ode.add(f'tau_ct_{i + 1}', value=0, min=1, max=50, vary=False)


if args.k1_shared == True:
    for y in range(num_rows - 1):
        params_ode[f'k1_{y + 2}'].expr = 'k1_1'

if args.k2_shared == True:
    for y in range(num_rows - 1):
        params_ode[f'k2_{y + 2}'].expr = 'k2_1'



if args.k1_fixed == True:
    params_ode[f'k1_1'].vary = False
    if args.k1 == 0:
        params_ode[f'k1_1'].min = 0
        params_ode[f'k1_1'].value = 0

if args.k2_fixed == True:
    params_ode[f'k2_1'].vary = False
    if args.k2 == 0:
        params_ode[f'k2_1'].min = 0
        params_ode[f'k2_1'].value = 0

if args.additional_early_decay == True:
    for i in range(num_rows):
        params_ode[f'A_ct_{i+1}'].vary = True
        params_ode[f'A_ct_{i+1}'].value = 0.2
        params_ode[f'tau_ct_{i+1}'].vary = True
        params_ode[f'tau_ct_{i+1}'].value = 5
    for y in range(num_rows - 1):
        params_ode[f'tau_ct_{y + 2}'].expr = 'tau_ct_1'

### Fitting the Data
results_Model2 = minimize(resid_global, params_ode, method='leastsq',
                          args=(time_fit, Data, limit, limit_inner, resid, 0), nan_policy='omit')



k1_val = []
k2_val = []
A_ct = []
tau_ct = []
R2_val = []

### Extract important Parameters from Fit
for i in range(num_rows):
    R2_val_calc, yfit1 = post_Fitting(results_Model2, Data[str(i)], limit, limit_inner, time_fit, empty_dat, i, nan_beg, nan_end)
    R2_val.append(R2_val_calc)
    Data[str('Fit_' + str(i))] = yfit1
    tau_ct.append(results_Model2.params[f'tau_ct_{i + 1}'].value)
    A_ct.append(results_Model2.params[f'A_ct_{i + 1}'].value)
    k1_val.append(results_Model2.params[f'k1_{i+1}'].value)
    k2_val.append(results_Model2.params[f'k2_{i + 1}'].value)

### k1k2 Lifetime(s)

k1k2_lifetime = 1/(np.array(k1_val)+np.array(Exc_Density)*np.array(k2_val)*(1-np.array(A_ct)))*1e9*0.5
k1k2_shared_lifetime = 1/(np.median(k1_val)+np.array(Exc_Density)*np.median(k2_val)*(1-np.array(A_ct)))*1e9*0.5

""" PLOTTING OF DATA """
### Define Plots
color1 = iter(cm.viridis(np.linspace(0, 1, (num_rows+1))))

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
ax2.set_xlabel('Excitation Density [cm⁻³]')


ax3 = fig.add_subplot(gs[1,2])
ax3.set_title('Other Parameters', fontsize = 14)
ax3.set_ylim(0,1)
ax3.set_ylabel('A-early time [-]')
ax3.set_xlabel('Excitation Density [cm⁻³]')



### Fill plots with content
for i in range(num_rows):
    color2 = next(color1)

    ax1.semilogy(Data['Time'], Data[str(i)], 'o', alpha=(0.25), c=color2,label=str(sample_name[i] + " (" + "{:.2f}".format(laser_fluence_old[i]) + "  nJ cm⁻²)"))
for i in range(num_rows):
    ax1.semilogy(Data['Time'], Data[str('Fit_' + str(i))], zorder=100, c='mediumvioletred', linewidth=3)


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
ax1.annotate(str("Wavelength: " +  "{:.0f}".format(wavelength) + " nm"),xy=(0.04,0.03),xycoords='axes fraction',c='black', fontsize=10)

ax1.text(10, 1e-5,str(" k1 constant: ~" + "{:.2e}".format(np.median(k1_val)) + " s⁻¹\n k2 constant: ~" + "{:.2e}".format(np.median(k2_val)) + " cm³ s⁻¹"),  fontsize=14,
        bbox=dict(boxstyle='round', facecolor=color2, alpha=0.25),c=color2)


ax1.legend()



ax2.semilogx(Exc_Density, e_Lifetime, marker='o', c='black', label="1/e Lifetime")
ax2.errorbar(Exc_Density, e_Lifetime, yerr=(err1, err2), c='black')

ax2.semilogx(Exc_Density, k1k2_lifetime, marker='o', c='mediumvioletred', label="k1k2 Lifetime (fits)")
ax2.semilogx(Exc_Density, k1k2_shared_lifetime, marker='o', c=color2 , label="k1k2 Lifetime (med.)")
ax2.legend()



ax3.semilogx(Exc_Density, A_ct, marker='o', c='lightblue',label='A-early times')

ax3.legend()



# Store Data and Parameters

Fit_Values = pd.DataFrame(index=[sample_name],columns=['Exc_Dens (cm-3)','k1 (s-1)','k2 (cm3s-1)','tau_aed (ns)','A_aed','tau-1/e (ns)','tau_k1k2 (ns)','pile-up (%)','R_sqr'])


for i in range(num_rows):
    Fit_Values.loc[sample_name[i],'Exc_Dens (cm-3)'] = str("{:.3e}".format(Exc_Density[i]))
    Fit_Values.loc[sample_name[i],'k1 (s-1)'] = str("{:.3e}".format(k1_val[i]))
    Fit_Values.loc[sample_name[i],'k2 (cm3s-1)'] = str("{:.3e}".format(k2_val[i]))
    Fit_Values.loc[sample_name[i], 'tau_aed (ns)'] = round(tau_ct[i],2)
    Fit_Values.loc[sample_name[i], 'A_aed'] = round(A_ct[i],2)

    Fit_Values.loc[sample_name[i], 'tau-1/e (ns)'] = round(e_Lifetime[i],1)
    Fit_Values.loc[sample_name[i], 'tau_k1k2 (ns)'] = round(k1k2_lifetime[i],1)
    Fit_Values.loc[sample_name[i], 'pile-up (%)'] = round(pile_up[i],1)
    Fit_Values.loc[sample_name[i], 'R_sqr'] = round(R2_val[i],3)

    Data = Data.rename({str(i): sample_name[i]}, axis='columns')
    Data = Data.rename({str('Fit_' + str(i)): str('Fit_' + sample_name[i])}, axis='columns')

Data = Data.replace(np.nan, str('NaN'), regex=True)


print('Fitting Results:')
print(Fit_Values.to_string())




### Saving the Data and Figure
Fit_Values.to_csv(str(PurePath(args.directory).joinpath(str(args.short_name).replace('.dat', '_fit-values_ode.txt'))), sep='\t', index= True, mode='w')
Data.to_csv(str(PurePath(args.directory).joinpath(str(args.short_name).replace('.dat', '_processed_ode.txt'))), sep='\t', mode='w')
plt.savefig(str(PurePath(args.directory).joinpath(str(args.short_name).replace('.dat', '_figure_ode.png'))), format='png',dpi=100)


plt.show()

exit()







def main(args, FileName):

    fig = plt.figure(figsize=(16, 9))
    plt.subplots_adjust(wspace=0.28, hspace=0.35)
    gs = gridspec.GridSpec(2, 3)
    fig.suptitle('Transient Photoluminescence (TCSPC) Fitting:    ' + sample_name[0]+" " +fit_proc , fontsize = 20)

    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('Photoluminescence Transients (Lin-Log)', fontsize = 14)
    ax1.set_xlim(-40, Fit_range[1]+100)
    ax1.set_ylim(1e-6,1.5)
    ax1.set_xlabel('Time after Pulse [ns]')
    ax1.set_ylabel('Normalized PL [a.u.]')

    ax6 = fig.add_subplot(gs[1,0])
    ax6.set_title('Photoluminescence Transients (Log-Log)', fontsize = 14)
    ax6.set_xlim(1, Fit_range[1]+100)
    ax6.set_ylim(1e-3,1.5)
    ax6.set_xlabel('Time after Pulse [ns]')
    ax6.set_ylabel('Normalized PL [a.u.]')

    ax4 = fig.add_subplot(gs[0,1])
    ax4.set_title('Differential Lifetimes of Fits', fontsize = 14)
    ax4.set_ylim(1,1000)
    ax4.set_xlabel('Time after Pulse [ns]')
    ax4.set_ylabel('Lifetime [ns]')

    ax2 = fig.add_subplot(gs[0, 2])             #   For Residual Analysis later

    ax5 = fig.add_subplot(gs[1,1])              #   For Lifetimes
    ax5.set_title('Lifetimes', fontsize = 14)
    ax5.set_xlim(np.amin(Exc_Density)/100,np.amax(Exc_Density)*10)
    ax5.set_ylabel('Lifetime [ns]')
    ax5.set_xlabel('Excitation Density [cm⁻³]')



    color1 = iter(cm.plasma(np.linspace(0, 1, (len(FileName)+1))))
    i = 0
    while i < len(FileName):
        color2 = next(color1)
        ax1.semilogy(Data['Time'], Data[str(i)], 'o',alpha= (0.25), c=color2, label=str("{:.2f}".format(laser_fluence_old[i]) + "  nJ cm⁻²"))
        ax6.loglog(Data['Time'], Data[str(i)], 'o',alpha= (0.25), c=color2, label=str("{:.2f}".format(laser_fluence_old[i]) + "  nJ cm⁻²"))

        i += 1


    ### Prepare for the Fitting(s)

    ## Find Fitting Range

    num_rows = len(FileName)
    time = np.array(Data['Time'])

    limit = np.where((time > -10) & (time < Fit_range[1]))
    nan_beg = np.empty(np.min(limit))
    nan_beg[:] = np.nan

    nan_end = np.empty(len(time)-np.amax(limit))
    nan_end[:] = np.nan



    time_fit = time[limit]*1e-9


    ## Define the three fitting limits
    limit0 = (Fit_range[1] - args.fit_start)/2
    limit0 = np.where((time_fit >= limit0*1e-9))[0][0]

    limit01 = (Fit_range[1] - args.fit_start)/3
    limit01 = np.where((time_fit >= limit01*1e-9))[0][0]

    limit1 = (Fit_range[1] - args.fit_start)/10
    limit1 = np.where((time_fit >= limit1*1e-9))[0][0]

    limit2 = np.where((time_fit >= args.fit_start*1e-9))[0][0]


    Fit_range[0] = time_fit[limit2]*1e9






    if args.fit_end <= 500:
        limit1 = np.where(time_fit > 1e-7)[0][0]
    else:
        limit1 = np.where(time_fit > 3e-7)[0][0]                # to fit k1 after 400 ns

    resid0 = np.empty(shape = (num_rows, len(time_fit)-limit0-1))
    resid01 = np.empty(shape = (num_rows, len(time_fit)-limit01-1))
    resid1 = np.empty(shape = (num_rows, len(time_fit)-limit1-1))
    resid2 = np.empty(shape = (num_rows, len(time_fit)-limit2-1))

    empty_dat = Data*0
    empty_dat[empty_dat == np.nan] = 0
    params_ode = Parameters()


    Fit_Values = pd.DataFrame(columns=[sample_name],index=['Exc_Dens','k1','k2ext','tau_aed','A_aed','tau_impl','tau_1e','R_sqr'])
    Fit_results = pd.DataFrame(columns=['value','q1','q3'],index={'k1_med','k2ext_med'})

    ### Defining the Parameters

    for i in range(num_rows):
        params_ode.add(f'k1_{i+1}', value=args.k1, min=1e5, max=1e10, vary=True)
        params_ode.add(f'k2_{i+1}', value=args.k2, min=1e-14, max=1e-3, vary=True)
        params_ode.add(f'k3_{i + 1}', value=28, min=25, max=29, vary=True)
        params_ode.add(f'A_ct_{i + 1}', value=0, min=0, max=1.000, vary=False)
        params_ode.add(f'tau_ct_{i + 1}', value=0, min=0, max=100e-9, vary=False)

    for y in range(num_rows-1):
        params_ode[f'k1_{y + 2}'].expr = 'k1_1'
        params_ode[f'k2_{y+2}'].expr = 'k2_1'
        params_ode[f'k3_{y + 2}'].expr = 'k3_1'
        params_ode[f'A_ct_{y + 2}'].expr = 'A_ct_1'
        params_ode[f'tau_ct_{y + 2}'].expr = 'tau_ct_1'

    if args.k1_fixed == True:
            params_ode[f'k1_1'].vary = False
            if args.k1 == 0:
                params_ode[f'k1_1'].min = 0
                params_ode[f'k1_1'].value = 0




    ### Fitting(s)


    ### First fit with focus on k1 ((fit_end-fit_start)/2  using only the differential equation)

    k1_fit = []

    for i in range(num_rows):
        results_Model2 = minimize(resid_global, params_ode, method='leastsq', args=(time_fit, Data, limit, limit0 , resid0, i),
                                  nan_policy='omit')

        k1_fit.append(results_Model2.params[f'k1_1'].value)



    ### Limiting the Range of k1
    params_ode[f'k1_1'].value = np.median(k1_fit)



    ### Fixing k2, if needed
    if args.k2_fixed == True:
            params_ode[f'k2_1'].vary = False



    ### Second Fit again with focus on k1, ((fit_end-fit_start)/3 possibly aed included)

    k1_fit = []

    for i in range(num_rows):
        results_Model2 = minimize(resid_global, params_ode, method='leastsq', args=(time_fit, Data, limit, limit01 , resid01, i),
                                  nan_policy='omit')


        k1_fit.append(results_Model2.params[f'k1_1'].value)


    ### Limiting the Range of k1
    params_ode[f'k1_1'].value = np.median(k1_fit)

    if np.min(k1_fit) != np.max(k1_fit):
        params_ode[f'k1_1'].min = np.median(k1_fit) * 0.1
        params_ode[f'k1_1'].max = np.median(k1_fit) * 10



    ### Activating post-correction aed, if used
    if args.additional_early_decay == True:
        for i in range(num_rows):
            params_ode[f'A_ct_{i+1}'].vary = True
            params_ode[f'A_ct_{i+1}'].value = 0.2
            params_ode[f'tau_ct_{i+1}'].vary = True
            params_ode[f'tau_ct_{i+1}'].value = 10e-9


    ## Third Fit ((fit_end-fit_start)/10

    k1_fit = []
    k2_fit = []
    for i in range(num_rows):
        results_Model2 = minimize(resid_global, params_ode, method='leastsq', args=(time_fit, Data, limit, limit1, resid1, i),
                                  nan_policy='omit')

        k2_fit.append(results_Model2.params[f'k2_1'].value)
        k1_fit.append(results_Model2.params[f'k1_1'].value)

    ### Limiting the range of k1

    params_ode[f'k1_1'].value = np.median(k1_fit)

    k2_fit = np.array(k2_fit)
    if np.max(k2_fit) > 1e-13:
        k2_fit[k2_fit < 1e-13] = np.nan
    params_ode[f'k2_1'].value = np.nanmedian(k2_fit)
    if np.min(k2_fit) != np.max(k2_fit):
        params_ode[f'k2_1'].min = np.nanmedian(k2_fit) * 0.01
        params_ode[f'k2_1'].max = np.nanmedian(k2_fit) * 10

    if np.min(k1_fit) != np.max(k1_fit):
        params_ode[f'k1_1'].min = np.median(k1_fit)*0.5
        params_ode[f'k1_1'].max = np.median(k1_fit)*2



    ### Fourth Fit (from fit_start to fit_end) using limited k1, k2 and if selected aed

    k1_fit = []
    tau_ct_fit = []
    A_ct_fit = []
    k2_fit = []
    R2_val = []
    tau_shared_k2 = []

    color1 = iter(cm.plasma(np.linspace(0, 1, (len(FileName)+1))))          # restart the iteration here
    for i in range(num_rows):
        results_Model2 = minimize(resid_global, params_ode, method='leastsq', args=(time_fit, Data, limit, limit2, resid2, i),
                                  nan_policy='omit')

        tau_ct_fit.append(results_Model2.params[f'tau_ct_1'].value * 1e9)
        A_ct_fit.append(results_Model2.params[f'A_ct_1'].value)
        k2_fit.append(results_Model2.params[f'k2_1'].value)
        k1_fit.append(results_Model2.params[f'k1_1'].value)
        yfit = -residual2(results_Model2.params, time_fit, empty_dat, limit, 0, i)  #

        ## Calculating R2 value of Fit
        Data_fit_test = np.array(Data[str(i)])[limit][:-1]
        residual = (Data_fit_test[limit2:]-yfit[limit2:])
        residual = np.nan_to_num(residual, nan=0)


        residual_norm = residual/np.amax(np.abs(residual))

        y2fit = np.nan_to_num(yfit, nan=0)
        R2_val_calc = 1-np.sum(residual**2)/np.sum((y2fit[limit2:]-np.mean(y2fit[limit2:]))**2)
        R2_val.append(R2_val_calc)

        ## Adjusting length of yfit array

        yfit = np.append(yfit, nan_end)
        yfit = np.append(nan_beg, yfit)


        ## 1/e Lifetime of Fit

        marker = np.where(yfit < 1 / np.e)[0]
        marker = np.array(Data['Time'])[marker]
        marker[marker <= 0] = np.nan
        marker = np.nanmin(marker)
        tau_shared_k2.append(marker)

        ## Store fit data
        Data[str('Fit_' + str(i))] = yfit

        ## Plot Fits in original graph

        ax1.semilogy(Data['Time'], yfit, zorder=100, c='mediumseagreen', linewidth=3)
        ax6.loglog(Data['Time'], yfit, zorder=100, c='mediumseagreen', linewidth=3)

        color2 = next(color1)
        yfit[yfit <=0] = np.nan
        diff_yfit = np.diff(-np.log(yfit))
        diff_yfit[diff_yfit == 0] = np.nan
        diff_yfit = 1/diff_yfit*((time_fit[2]-time_fit[1])*1e9)
        diff_yfit = np.append(diff_yfit,0)
        diff_yfit[diff_yfit <= 0] = np.nan
        ax4.loglog(np.array(Data['Time'])[np.where(Data['Time']>0)], diff_yfit[np.where(Data['Time']>0)],
                   linewidth=3, c=color2, alpha = 0.5, label=str("{:.2f}".format(laser_fluence_old[i]) + "  nJ cm⁻²  Fit"))



        ## Residual Plot
        ax2.plot(time_fit[limit2:-1] * 1e9, residual_norm,
                 c=color2,alpha=0.4,zorder= -20*i)




    ### Saving Data of Final Fit

    Fit_results['value']['k1_med'] = np.median(k1_fit)

    k2_fit = np.array(k2_fit)

    if np.max(k2_fit) > 1e-13:
        k2_fit[k2_fit < 1e-13] = np.nan

    k2_fit_value = np.nanmedian(k2_fit)
    Fit_results['value']['k2ext_med'] = "{:.2e}".format(k2_fit_value)


    if num_rows == 1 or params_ode[f'k1_1'].vary == False:
        Fit_results['q1']['k1_med'] = np.nan

    if num_rows == 1 or params_ode[f'k2_1'].vary == False:
        Fit_results['q1']['k2ext_med'] = np.nan

    else:
        Fit_results['q1']['k1_med'] = str("{:.2e}".format(abs(np.quantile(k1_fit,0.25) - np.median(k1_fit))))
        Fit_results['q3']['k1_med'] = str("{:.2e}".format(abs(np.quantile(k1_fit, 0.75) - np.median(k1_fit))))
        Fit_results['q1']['k2ext_med'] = str("{:.2e}".format(abs(np.quantile(k2_fit, 0.25) - np.median(k2_fit))))
        Fit_results['q3']['k2ext_med'] = str("{:.2e}".format(abs(np.quantile(k2_fit, 0.75) - np.median(k2_fit))))









    ## Plotting implied tau from the fitted k1 and k2

    tau_shared_k = 1 / (Fit_results['value']['k1_med'] + np.array(Exc_Density)*k2_fit_value)*0.5 * 1e9


    tau_shared_k = ((tau_shared_k*(1-np.array(A_ct_fit)))**2+(np.array(A_ct_fit)*np.array(tau_ct_fit))**2)/((tau_shared_k*(1-np.array(A_ct_fit)))+np.array(A_ct_fit)*np.array(tau_ct_fit))


    ax5.semilogx(Exc_Density, tau_shared_k, c='darkblue', marker='o', label="from Fit constants")
    ax5.semilogx(Exc_Density,tau_shared_k2,c='mediumseagreen',marker='o',label="1/e Lifetime Fit")



    ax2.set_xlim(-40,args.fit_end + 100)
    ax2.set_ylim(-1,1)
    ax2.set_xlabel('Time after Pulse [ns]')
    ax2.set_ylabel('Residual norm. (a.u.; weighting: √t)')
    ax2.set_title('Fit Residuals', fontsize = 14)
    ax2.hlines(xmin=0, xmax = args.fit_end+100, y=0,  color='black')


    ### Show Fitting Range on Graph
    ax1.vlines(x=[Fit_range[0],Fit_range[1]], ymin=1e-6, ymax=1.5, color='grey',linestyles='--')
    ax1.annotate(
            'Fitting Range',
            xy=(0.7, 0.02), xycoords='axes fraction',c='grey')




    ### Sorting and Renaming the Data


    i = 0
    while i < num_rows:
        column_name = sample_name[i]
        Fit_Values.loc['Exc_Dens',column_name] = Exc_Density[i]
        Fit_Values.loc['k1',column_name] = k1_fit[i]
        Fit_Values.loc['k2ext',column_name] = "{:.3e}".format(k2_fit[i])
        Fit_Values.loc['tau_aed',column_name] = tau_ct_fit[i]
        Fit_Values.loc['A_aed',column_name] = A_ct_fit[i]
        Fit_Values.loc['tau_impl',column_name] = tau_shared_k2[i]
        Fit_Values.loc['tau_1e',column_name] = e_Lifetime[i]
        Fit_Values.loc['R_sqr', column_name] = R2_val[i]
        i += 1





    ### Showing the Fitting Results

    ax3 = fig.add_subplot(gs[1,2])
    ax3.text(0.0,.9,str('Fitting Results'),
             c='black', fontsize = 14)
    ax3.text(0.0,0.8,str('k1 constant (med)'),
             c='navy', fontsize = 10)
    if args.k1_fixed == True:
        ax3.text(0.7, 0.8, str("{:.2e}".format(results_Model2.params[f'k1_1'].value) +  " s⁻¹  (fixed)  "),
                 c='mediumseagreen', fontsize=10)
    elif params_ode[f'k1_1'].vary == False or num_rows == 1:
        ax3.text(0.7, 0.8, str("{:.2e}".format(results_Model2.params[f'k1_1'].value) + " s⁻¹"),
                 c='mediumseagreen', fontsize=10)
    else:
        ax3.text(0.7, 0.8, str("{:.2e}".format(Fit_results['value']['k1_med']) + "  +/-  " + str(Fit_results['q1']['k1_med']) + " s⁻¹"),
                 c='mediumseagreen', fontsize=10)


    ax3.text(0.0,0.7,str('k2 constant (med.)'),
             c='navy', fontsize = 10)
    if args.k2_fixed == True:
        ax3.text(0.7, 0.7, str("{:.1e}".format(results_Model2.params[f'k2_1'].value) + " cm³ s⁻¹  (fixed) "),
                 c='mediumseagreen', fontsize=10)
    elif params_ode[f'k2_1'].vary == False or num_rows == 1:
        ax3.text(0.7, 0.7, str("{:.2e}".format(results_Model2.params[f'k2_1'].value) + " cm³ s⁻¹"),
                 c='mediumseagreen', fontsize=10)
    else:
        ax3.text(0.7, 0.7, str(Fit_results['value']['k2ext_med'] + "  +/-  " + Fit_results['q1']['k2ext_med'] + " cm³ s⁻¹"),
                 c='mediumseagreen', fontsize=10)


    if args.additional_early_decay == True:
        ax3.text(0.0, 0.5, str('Additional Early Decay (AED)'),
                 c='navy', fontsize=10)
        ax3.text(0.7, 0.5, str("< " + "{:.1f}".format(np.amax(tau_ct_fit)) + " ns "),
                 c='mediumseagreen', fontsize=10)
        ax3.text(0.0, 0.4, str('Corr. Factor for AED (A)'),
                 c='navy', fontsize=10)
        ax3.text(0.7, 0.4, str("< " + "{:.1f}".format(np.amax(A_ct_fit)*100)+"%"),
                 c='mediumseagreen', fontsize=10)



    ax3.text(0.0,0.25,str('Pile-up Rate: ' ),c='black', fontsize = 10)
    if pile_up < 5.0:
        ax3.text(0.30, 0.25, str("≤ " + "{:.1f}".format(pile_up) + "%"), c='green', fontsize=10)
    elif (pile_up > 5.0) & (pile_up < 8.0):
        ax3.text(0.30, 0.25, str("≤ " + "{:.1f}".format(pile_up) + "%  (should be mentioned)"), c='orange', fontsize=10)
    else:
        ax3.text(0.30, 0.25, str("≤ " + "{:.1f}".format(pile_up) + "%  (data probably affected)"), c='red', fontsize=10)


    ax3.text(0.0,0.15,str('R² value: '),
             c='black', fontsize = 10)
    ax3.text(0.30,0.15,str("{:.3f}".format(np.mean(R2_val)) + "  +/-  " + "{:.3f}".format(np.std(R2_val))),
             c='black', fontsize = 10)



    ax3.text(0.0,0.05,str('Fit from: '),
             c='black', fontsize = 10)
    ax3.text(0.30,0.05,str("{:.0f}".format(Fit_range[0]) + "  to  " + "{:.0f}".format(Fit_range[1])+ " ns"),
             c='black', fontsize = 10)
    ax3.axis('off')



    ### Rename and Save Data and Fitting Data

    i = 0
    while i < num_rows:

        Data = Data.rename({str(i): sample_name[i]},axis='columns')
        Data = Data.rename({str('Fit_'+str(i)): str('Fit_'+sample_name[i])},axis='columns')

        i += 1

    ax1.legend()
    ax4.legend()
    ax5.legend()

    ### Saving the Data and Figure
    Fit_Values.to_csv(str(PurePath(args.directory).joinpath(str(args.short_name).replace('.dat', '_fit-values.txt'))), sep='\t', index= True, mode='w')
    Fit_results.to_csv(str(PurePath(args.directory).joinpath(str(args.short_name).replace('.dat', '_fit-values.txt'))), sep='\t', index= True, mode='a')
    Data.to_csv(str(PurePath(args.directory).joinpath(str(args.short_name).replace('.dat', '_processed.txt'))), sep='\t', mode='w')
    plt.savefig(str(PurePath(args.directory).joinpath(str(args.short_name).replace('.dat', '.pdf'))), format='pdf', dpi=100)

    ### Check, which Fitting Procedure was used



    print('\nFitting Results for: ' + str(sample_name[0])+'\n('+fit_proc+') \n =================')
    print(Fit_results)
    print('R² est. = ' + str("{:.3f}".format(np.mean(R2_val))))

    if args.Combined_Fit == True:
        plt.show()




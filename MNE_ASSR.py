
import numpy as np
import scipy as sp
import pandas as pd
import mne
import matplotlib.pyplot as plt

from pyfftw.interfaces import scipy_fft as fftw
from scipy.stats import mode
from matplotlib.lines import Line2D


from Trig_decoder_PHL import *

## convert units (to be used in mne apply function)

#convert to volt (MNE uses volts)
def uVtoV(data):
    return data*(10**(-6))

#mV to V
def mVtoV(data):
    return data*(10**(-3))

#convert from volt to nV
def VtonV(data):
    return data*(10**(9))

#convert to dB over 1uV
def in_dB_over_1uV(vector, power = False):
    if power:
        result = 10*np.log10(vector/10**(-12))
    else:
        result = 20*np.log10(vector/10**(-6))
    return result

#convert to dB over 1nV
def in_dB_over_1nV(vector, power = False):
    if power:
        result = 10*np.log10(vector/10**(-18))
    else:
        result = 20*np.log10(vector/10**(-9))
    return result

#root mean squared (It does not exists?????)
def rms(vector):
    N = len(vector)
    sumsquared = 0
    for i in vector:
        sumsquared+=i**2
    rms = np.sqrt(sumsquared/N)
    return rms




## ASSR specific functions

# ASSR analysis functions, it accepts a power spectrum from mne, modulation frequency, lower and upper frequency limits for noise estimation, significance level and tagging frequency tolerance (lower it if you get more than one modulation frequency point, increase it if you get no points).
# It returns a pandas dataframe with the amplitude, noise, SNR, p-value and significance for each channel.

def ASSR_Analysis(powerspectrum, AMfreq, deltaf_low, deltaf_high, significance = 0.01, tagging_frequency_tollerance = 0.1, power = True):
    freqs = powerspectrum.freqs
    ch_good = [x for x in powerspectrum.info["ch_names"] if x not in powerspectrum.info["bads"]]

    idxSignal =  np.where(np.abs(freqs-AMfreq)<=tagging_frequency_tollerance)
    if len(idxSignal[0]) == 0:
        raise ValueError('No signal frequency found, increase tagging_frequency_tollerance')
    elif len(idxSignal[0]) > 1:
        raise ValueError('More than one signal frequency found, decrease tagging_frequency_tollerance')
    else:
        idxSignal = idxSignal[0]
    signal = powerspectrum.get_data()[:,idxSignal][:,0]
    signalindB = in_dB_over_1nV(powerspectrum.get_data()[:,idxSignal][:,0], power=power)

    idxNoise = np.intersect1d(np.where(AMfreq-deltaf_low<=freqs)[0],np.where(AMfreq+deltaf_high>=freqs)[0])
    idxNoise = np.intersect1d(idxNoise, np.where(np.abs(freqs-AMfreq)>tagging_frequency_tollerance)[0])

    noise = []
    for ch_data in powerspectrum.get_data()[:,idxNoise]:
        noise.append(rms(ch_data))
    noise = np.array(noise)

    noiseindB = []
    for ch_data in powerspectrum.get_data()[:,idxNoise]:
        noiseindB.append(in_dB_over_1nV(rms(ch_data), power=power))
    noiseindB = np.array(noiseindB)
    

    
    N = len(idxNoise)
    f_value = np.power(signal,2)/np.power(noise,2)
    #print(f_value)
    snr = signal/noise
    snrindB = signalindB-noiseindB

    df1 = 2
    df2 = 2*N
    p = 1-sp.stats.f.cdf(f_value, df1, df2)
    significant = p<significance
    data = {'ampdB': signalindB, 'noisedB': noiseindB, 'snrdB': snrindB, 'p': p, 'significant':significant, 'f_value': f_value, 'signal': signal, 'noise': noise, 'snr': snr}
    output = pd.DataFrame(data, index=ch_good)
    return output


# ASSR plotting functions, it accepts a power spectrum from mne, modulation frequency, lower and upper frequency limits for noise estimation, significance level, tagging frequency tolerance (lower it if you get more than one modulation frequency point, increase it if you get no points) and a bunch of plotting options
# It plots the power spectrum of each channel with the modulation frequency point and the noise level, it also shows the SNR and significance of the modulation frequency point.

def ASSR_Plot(powerspectrum, AMfreq, deltaf_low, deltaf_high, plotsperrow, title, channels= None, significance = 0.01, save = False, show = True,
               ylim = (-35,0), xlim = (18,100), tickintervalx = 20, tickintervaly = 10, grid = False, tagging_frequency_tollerance =0.1,
                fig_height = 4.8, fig_width = 6.4, filepath = '', power = True):
    
    #if no channel list given, use all good channels
    if channels == None:
        channels = [x for x in powerspectrum.info["ch_names"] if x not in powerspectrum.info["bads"]]

    subplot_kw = dict(ylim = ylim, xlim = xlim, xticks = np.arange((xlim[0]//tickintervalx+1)*(tickintervalx), xlim[1]+1,tickintervalx), xlabel = 'frequency (Hz)',
                       ylabel = r'amplitude (dB 1$\mu$V)', yticks = np.arange((ylim[0]//tickintervaly+1)*(tickintervaly),ylim[1]+1,tickintervaly))
    fig, axs = plt.subplots(int(len(channels)//(plotsperrow+0.1)+1), plotsperrow, sharey=True, sharex = True, subplot_kw=subplot_kw)
    axs = np.array(axs)
    axs = np.reshape(axs, (int(len(channels)//(plotsperrow+0.1)+1), plotsperrow))
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)

    analysis = ASSR_Analysis(powerspectrum, AMfreq, deltaf_low, deltaf_high, significance, tagging_frequency_tollerance, power = power)
    
    for i, ch in enumerate(channels):
        ix = i//plotsperrow
        iy = i%plotsperrow
        marker = '.' if analysis.significant[ch] else 'x'
        axs[ix][iy].plot(powerspectrum.freqs, in_dB_over_1nV(np.transpose(powerspectrum.get_data(picks = [ch])), power=power))
        axs[ix][iy].set_title(ch+" (SNR = " + str(round(analysis.snrdB[ch],1))+" dB)")
        axs[ix][iy].scatter(AMfreq,analysis.ampdB[ch], c = 'red', marker = marker)
        axs[ix][iy].scatter(xlim[0],analysis.ampdB[ch], c = 'red', marker = "_", s = 150)
        axs[ix][iy].scatter(xlim[0],analysis.noisedB[ch], c = 'g', marker = "_", s = 150)
        axs[ix][iy].hlines(analysis.noisedB[ch], AMfreq-deltaf_low, AMfreq+deltaf_high, colors = 'g')
        axs[ix][iy].legend(handles=[ Line2D([0], [0], marker=marker, lw =0, color='r',markerfacecolor='r', markersize = 5, label=f'Signal= \n{round(analysis.ampdB[ch],1)}dB'),Line2D([0], [0], color='g', label=f'Noise= \n{round(analysis.noisedB[ch],1)}dB')], loc='upper right')
        if grid:
            axs[ix][iy].grid()
    fig.suptitle(title)
    if save:
        fig.savefig(filepath+title+".png")
    if show:
        fig.show()

# Same as the two functions above but for a powerspectrum vector (e.g. the average of the powerspectrum of all channels)

def get_spectrum_ASSR_analysis(powerspectrum, freqs, AMfreq, deltaf_low, deltaf_high,significance = 0.01, tagging_frequency_tollerance =0.1, power = True):
    idxSignal =  np.where(np.abs(freqs-AMfreq)<=tagging_frequency_tollerance)[0]
    signal = powerspectrum[idxSignal]
    if len(signal) == 0:
        raise ValueError('No signal found in the given frequency range, consider increasing the tagging_frequency_tollerance')
    elif len(signal) > 1:
        raise ValueError('Multiple signals found in the given frequency range, consider decreasing the tagging_frequency_tollerance')
    else:
        signal = signal[0]

    signalindB = in_dB_over_1nV(signal, power=power)

    idxNoise = np.intersect1d(np.where(AMfreq-deltaf_low<=freqs)[0],np.where(AMfreq+deltaf_high>=freqs)[0])
    idxNoise = np.intersect1d(idxNoise, np.where(np.abs(freqs-AMfreq)>tagging_frequency_tollerance)[0])


    noise = rms(powerspectrum[idxNoise])
    noiseindB = in_dB_over_1nV(noise, power=power)
    

    f_value = np.power(signal,2)/np.power(noise,2)
    
    snr = signal/noise
    snrindB = signalindB-noiseindB

    N = len(idxNoise)

    df1 = 2
    df2 = 2*N
    p = 1-sp.stats.f.cdf(f_value, df1, df2)
    significant = p<significance
    data = {'ampdB': signalindB, 'noisedB': noiseindB, 'snrdB': snrindB, 'p': p, 'significant':significant}

    return data



def Plot_average_spectra(powerspectrum, AMfreq, deltaf_low, deltaf_high, title = '', channels= None, significance = 0.01, save = False, show = True,
               ylim = (-35,0), xlim = (18,100), tickintervalx = 20, tickintervaly = 10, grid = False, tagging_frequency_tollerance =0.1,
                fig_height = 4.8, fig_width = 6.4, filepath = '', colour = 'b'):
    if channels == None:
        channels = [x for x in powerspectrum.info["ch_names"] if x not in powerspectrum.info["bads"]]

    axes_kw = dict(ylim = ylim, xlim = xlim, xticks = np.arange((xlim[0]//tickintervalx+1)*(tickintervalx), xlim[1]+1,tickintervalx), xlabel = 'frequency (Hz)',
                       ylabel = r'amplitude (dB 1nV)', yticks = np.arange((ylim[0]//tickintervaly+1)*(tickintervaly),ylim[1]+1,tickintervaly))
    average = np.average(powerspectrum.get_data(picks = channels),0)

    analysis = get_spectrum_ASSR_analysis(average, powerspectrum.freqs,  AMfreq, deltaf_low, deltaf_high, significance, tagging_frequency_tollerance)
    marker = '.' if analysis['significant'] else 'x'
    fig, ax= plt.subplots()
    ax.set(**axes_kw)
    for ch in channels:
        ax.plot(powerspectrum.freqs,np.transpose(in_dB_over_1nV(powerspectrum.get_data(picks = ch), power=True)),linestyle='dashed', c = colour, linewidth = 0.5, alpha = 0.5)
    ax.plot(powerspectrum.freqs,in_dB_over_1nV(average, power=True), c = colour)
    ax.scatter(AMfreq,analysis['ampdB'], c = 'red', marker = marker, s = 100)
    ax.scatter(xlim[0],analysis['ampdB'], c = 'red', marker = "_", s = 150)
    ax.scatter(xlim[0],analysis['noisedB'], c = 'g', marker = "_", s = 150)
    ax.hlines(analysis['noisedB'], AMfreq-deltaf_low, AMfreq+deltaf_high, colors = 'g')
    if show:
        fig.show()

        
# Returns a pandas dataframe with the selected property of the modulation frequency point of each channel referenced to other channels, average and specified channel groups.
def get_reference_overview(evoked, AMfreq, deltaf_low, deltaf_high, property = 'snrdB', tagging_frequency_tollerance = 0.1, channel_groups = dict(), method = 'fft'):

    evoked_copy = evoked.copy()

    df = pd.DataFrame(index=good_channels(evoked_copy)+['average']+list(channel_groups.keys()), columns=good_channels(evoked_copy)+['average']+list(channel_groups.keys()))

    if method == 'fft':

        for ch in good_channels(evoked_copy):
            evoked_copy.set_eeg_reference(ref_channels=[ch])
            data = ASSR_Analysis(fft(evoked_copy), AMfreq, deltaf_low, deltaf_high, tagging_frequency_tollerance = tagging_frequency_tollerance, power=False)
            for ch2 in good_channels(evoked_copy):
                df.at[ch,ch2] = data.at[ch2,property]

        for name, group in channel_groups.items():
            evoked_copy.set_eeg_reference(ref_channels=list(np.intersect1d(group, good_channels(evoked_copy))))
            data = ASSR_Analysis(fft(evoked_copy), AMfreq, deltaf_low, deltaf_high, tagging_frequency_tollerance = tagging_frequency_tollerance, power=False)
            for ch2 in good_channels(evoked_copy):
                        df.at[name,ch2] = data.at[ch2,property]


        evoked_copy.set_eeg_reference(ref_channels='average')
        data = ASSR_Analysis(fft(evoked_copy), AMfreq, deltaf_low, deltaf_high, tagging_frequency_tollerance = tagging_frequency_tollerance, power=False)
        for ch2 in good_channels(evoked_copy):
                df.at['average',ch2] = data.at[ch2,property]

    elif method == 'welch':

        for ch in good_channels(evoked_copy):
            evoked_copy.set_eeg_reference(ref_channels=[ch])
            data = ASSR_Analysis(evoked_copy.compute_psd(fmin = 10, fmax = 100, method = 'welch',window = 'boxcar', remove_dc = True), AMfreq, deltaf_low, deltaf_high, tagging_frequency_tollerance = tagging_frequency_tollerance)
            for ch2 in good_channels(evoked_copy):
                df.at[ch,ch2] = data.at[ch2,property]

        for name, group in channel_groups.items():
            evoked_copy.set_eeg_reference(ref_channels=list(np.intersect1d(group, good_channels(evoked_copy))))
            data = ASSR_Analysis(evoked_copy.compute_psd(fmin = 10, fmax = 100, method = 'welch',window = 'boxcar', remove_dc = True), AMfreq, deltaf_low, deltaf_high, tagging_frequency_tollerance = tagging_frequency_tollerance)
            for ch2 in good_channels(evoked_copy):
                        df.at[name,ch2] = data.at[ch2,property]


        evoked_copy.set_eeg_reference(ref_channels='average')
        data = ASSR_Analysis(evoked_copy.compute_psd(fmin = 10, fmax = 100, method = 'welch',window = 'boxcar', remove_dc = True), AMfreq, deltaf_low, deltaf_high, tagging_frequency_tollerance = tagging_frequency_tollerance)
        for ch2 in good_channels(evoked_copy):
                df.at['average',ch2] = data.at[ch2,property]

    else:

        print('method '+ method + ' not yet implemented')

    return df

#Plots the reference overview dataframe in a heatmap with the specified colormap, colorbar label, threshold for black and white text (adjust according to colourmap), title and tagging frequency tolerance.
def plot_reference_summary(evoked, AMfreq, deltaf_low, deltaf_high, cmap = plt.cm.Blues, cbar_label = 'dB', bwthres = 0.75, title = None, tagging_frequency_tollerance = 0.1,
                            channel_groups = dict(), method = 'fft', property = 'snrdB'):
    df = get_reference_overview(evoked, AMfreq, deltaf_low, deltaf_high, tagging_frequency_tollerance = tagging_frequency_tollerance, channel_groups = channel_groups, method = method, property = property)
    df_round = df[good_channels(evoked)].replace(-np.inf, -np.Inf).replace(np.nan, -np.Inf).round(decimals=2)
    df_round.replace(-np.Inf, np.nan, inplace = True)
    df_plot = df_round.replace(np.nan, "")

    if property == 'snrdB':
        norm = plt.Normalize(5, 30, clip = True)
    elif property == 'ampdB':
        norm = plt.Normalize(30, 60, clip = True)
    elif property == 'noisedB':
        norm = plt.Normalize(20, 40, clip = True)
    else:
        norm = plt.Normalize(0, 100, clip = True)

    data = df_round.values
    data_plot = df_plot.values

    reference = df_round.index
    electrode = df_round.columns

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap = cmap, norm = norm, aspect='auto')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(electrode)), labels=electrode)
    ax.set_yticks(np.arange(len(reference)), labels=reference)
    ax.set_xlabel("electrode")
    ax.set_ylabel("reference")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")


    for i in range(len(reference)):
        for j in range(len(electrode)):
            if norm(data)[i][j] >= bwthres:
                ax.text(j, i, data_plot[i, j],
                            ha="center", va="center", color="w")
            else:
                ax.text(j, i, data_plot[i, j],
                            ha="center", va="center", color="k")

    if title == None:
        if property == 'snrdB':
            title = 'SNR'
        elif property == 'ampdB':
            title = 'Signal'
        elif property == 'noisedB':
            title = 'Noise floor'
        else:
            title = ''


    ax.set_title(title)
    fig.tight_layout()
    plt.show()

#takes a list of spectrum objects (one per subject) and returns the grand average spectrum
def grand_average_spectrum(spectrum_list):
    averages = []
    for subject_spectrum in spectrum_list:
        ch_good = [x for x in subject_spectrum.info["ch_names"] if x not in subject_spectrum.info["bads"]]
        averages.append(np.average(subject_spectrum.get_data(picks = ch_good),0))
    grand_average = np.average(np.array(averages),0)
    return grand_average

#Plots the grand average spectrum of a list of spectrum objects (one per subject) with the average for all channels for each subject in the background.
def plot_grand_average_spectra_avgs_only(spectrum_list, AMfreq, deltaf_low, deltaf_high, title = '', channels= None, significance = 0.01, save = False, show = True,
               ylim = (-35,0), xlim = (18,100), tickintervalx = 20, tickintervaly = 10, grid = False, tagging_frequency_tollerance =0.1,
                fig_height = None, fig_width = None, filepath = '', colourlist = ['b', 'g', 'r', 'm', 'c']):
    axes_kw = dict(ylim = ylim, xlim = xlim, xticks = np.arange((xlim[0]//tickintervalx+1)*(tickintervalx), xlim[1]+1,tickintervalx), xlabel = 'frequency (Hz)',
                       ylabel = r'amplitude (dB 1nV)', yticks = np.arange((ylim[0]//tickintervaly+1)*(tickintervaly),ylim[1]+1,tickintervaly)) 
    averages = []
    fig, ax= plt.subplots()
    if fig_height != None:
        fig.set_figheight(fig_height)
    
    if fig_width != None:
        fig.set_figwidth(fig_width)

    ax.set(**axes_kw)
    for i, subject_spectrum in enumerate(spectrum_list):
        if channels == None:
            channels = [x for x in subject_spectrum.info["ch_names"] if x not in subject_spectrum.info["bads"]]
        ch_good = [x for x in channels if x not in subject_spectrum.info["bads"]]
        average = np.average(subject_spectrum.get_data(picks = ch_good),0)
        averages.append(average)
        ax.plot(subject_spectrum.freqs,in_dB_over_1nV(average, power=True), c = colourlist[i], linewidth = 0.5, alpha = 0.3)
    grand_average = np.average(np.array(averages),0)
    analysis = get_spectrum_ASSR_analysis(grand_average, spectrum_list[0].freqs,  AMfreq, deltaf_low, deltaf_high, significance, tagging_frequency_tollerance)
    marker = '.' if analysis['significant'] else 'x'
    ax.plot(spectrum_list[0].freqs,in_dB_over_1nV(grand_average, power=True), c = 'k')
    ax.scatter(AMfreq,analysis['ampdB'], c = 'red', marker = marker, s = 100)
    ax.scatter(xlim[0],analysis['ampdB'], c = 'red', marker = "_", s = 150)
    ax.scatter(xlim[0],analysis['noisedB'], c = 'g', marker = "_", s = 150)
    ax.hlines(analysis['noisedB'], AMfreq-deltaf_low, AMfreq+deltaf_high, colors = 'g')
    ax.legend(handles=[Line2D([0], [0], marker=marker, lw =0, color='r',markerfacecolor='r', markersize = 5, label='Signal= '+str(round(analysis['ampdB'],1))+'dB'),
                       Line2D([0], [0], color='g', label='Noise= '+str(round(analysis['noisedB'],1))+ 'dB')], loc='upper right')

    fig.suptitle(title + ' (SNR= '+str(round(analysis['snrdB'],1))+ 'dB' + ' p= '+str(round(analysis['p'],5))+')')
    if save:
        fig.savefig(filepath+title+".png")
    if show:
        plt.show()
## Triggers and trigger debugging functions

#use the decoder to get the triggers from the raw data. Trig channel should be Dig type.
def get_trigs(raw, TRIGGER_CLK = 18, THR_ERROR = -0.05, TRANS_ERROR = 0.1):
    info = raw.info
    trig_data = np.logical_not((raw.get_data(picks='Dig')[0]>0)).astype(int)
    decoder = SerialTriggerDecoder(trig_data, info['sfreq'], TRIGGER_CLK, THR_ERROR, TRANS_ERROR)
    decoded_trigs = decoder.decode()
    return decoded_trigs

#get an MNE event list from the decoder object
def events_from_decoder(decoded_trigs):
    events = np.empty((len(decoded_trigs),3), int)
    for i, trig in enumerate(decoded_trigs):
        event = [trig['sample_idx'], 0, trig['code']]
        events[i] = event
    return events

#gives you the list of time differences between triggers
def get_trig_distances(decoded_trigs):
    trig_diff = []
    for i in range(1, len(decoded_trigs)):
        trig_diff.append(decoded_trigs[i]['sample_idx']-decoded_trigs[i-1]['sample_idx'])
    return trig_diff

#checks wether two trigger lists are coincident. Mostly useful when aligning audios.
def trig_coincidence(decoded_trigs_1, decoded_trigs_2, tollerance = 0):
    coincident = []
    for i, trig in enumerate(decoded_trigs_1):
        coincident.append(abs(trig['sample_idx']-decoded_trigs_2[i]['sample_idx'])<=tollerance)
    return coincident

#plot an histogram of the time differences between two trigger lists for debugging purposes
def trigger_lag_histo(trigs1, trigs2):
    diff =[]
    for i in range(1, len(trigs1)):
        diff.append(abs(trigs1[i]['sample_idx'] - trigs1[i-1]['sample_idx'] -(trigs2[i]['sample_idx'] - trigs2[i-1]['sample_idx'])))

    bins =plt.hist(diff,range=(0.5,100.5), bins = 100)[1]
    ax = plt.gca()
    ax.set_xticks(bins - 0.5)
    plt.show()
    return diff

#return the time differences between two trigger lists for debugging purposes
def trigger_lags(trigs1, trigs2):
    diff =[]
    for i in range(1, len(trigs1)):
        diff.append(trigs1[i]['sample_idx'] - trigs1[i-1]['sample_idx'] -(trigs2[i]['sample_idx'] - trigs2[i-1]['sample_idx']))
    return diff

#get unique codes from a trigger list repeating at least min_trigs_number times
def get_repeted_codes(decoded_trigs, min_trigs_number = 5):
    codes = []
    for code in np.unique([trigs['code'] for trigs in decoded_trigs]):
        if np.sum(np.where(np.array([trigs['code'] for trigs in decoded_trigs]) == code, 1, 0)) >= min_trigs_number:
            codes.append(code)
    return codes


## Creating and modifying event lists

#add intermediate events to an event list. It is useful when you have a long epoch and you want to split it in smaller epochs.
def add_intermediate_events(events, new_epoch_length, sfreq):
    events_new = []
    epoch_length = events[2][0]-events[1][0]
    samples_per_epoch = int(sfreq*new_epoch_length)
    events_per_epoch = int(np.floor(epoch_length/samples_per_epoch))
    samples_per_epoch = int(sfreq*new_epoch_length)
    for event in events:
        for i in range(0, events_per_epoch):
            events_new.append([event[0]+(i*samples_per_epoch), event[1], event[2]])
    events_new = np.array(events_new)
    return(events_new)

#creates a list of equally spaced events for a raw object
def create_equally_spaced_events(raw, length, code = 1):
    events = []
    samples_per_epoch = int(length*raw.info['sfreq'])
    epoch_number = int(raw.n_times//samples_per_epoch)
    for i in range(0,epoch_number):
        events.append(([i*samples_per_epoch, 0, code]))

    events = np.array(events)
    return events


##General functions

#crop the raw object based on a list of events with a specific code (will be reworked, I would epoch and use concatenate_epochs_raw instead)
def code_based_crop(raw, events, code, copy = True):
    samples = [event[0] for event in events if event[2] == code]
    samplestrasl1 = [0]+samples[0:-1]
    samples = np.array(samples)
    samplestrasl1 = np.array(samplestrasl1)

    diff = samples-samplestrasl1
    diff = diff[1:]

    min_samp = np.min(samples)
    max_samp = np.max(samples)
    epoch_lenght = mode(diff).mode

    tmin = min_samp/raw.info['sfreq']
    tmax = (max_samp+epoch_lenght)/raw.info['sfreq']

    if copy:
        cropped_data = raw.copy().crop(tmin = tmin, tmax = tmax)
    else:
        cropped_data = raw.crop(tmin = tmin, tmax = tmax)

    return cropped_data

#concatenate epochs back in a raw object, useful when selecting only part of the experiment.
def concatenate_epochs_raw(epochs):
    epochs_data = epochs.get_data()
    reshaped_epochs = np.zeros((epochs_data.shape[0]*epochs_data.shape[2],epochs_data.shape[1]))
    for i in range(0, epochs_data.shape[0]):
        for j in range(0, epochs_data.shape[1]):
            for k in range(0, epochs_data.shape[2]):
                reshaped_epochs[i*epochs_data.shape[2]+k,j] = epochs_data[i][j][k]
    reshaped_epochs = reshaped_epochs.transpose()
    new_raw = mne.io.RawArray(reshaped_epochs, epochs.info, first_samp=0, copy='auto', verbose=None)
    return new_raw

#Takes and mne Epochs object and returns the average of the epochs weighted by the weights given in the weights list
def weighted_average(epochs, weights):
    epochs_data = epochs.get_data()
    weights_repeated = np.zeros(epochs_data.shape)
    for i, epoch_w in enumerate(weights):
        for j, ch_w in enumerate(epoch_w):
            weights_repeated[i][j] = np.repeat(ch_w, epochs_data.shape[2])
    weighted_avg = np.average(epochs_data, axis=0, weights = weights_repeated)
    evoked_weighted_avg = mne.EvokedArray(weighted_avg, epochs.info, tmin=0.0, nave=epochs_data.shape[0])
    return evoked_weighted_avg

#weights the epochs based on the variance of the data in each epoch and returns the weighted average
def variance_weighted_average(epochs):
    epochs_data = epochs.get_data()
    variances = np.var(epochs_data, axis=2)
    weights = 1/variances
    evoked_weighted_avg = weighted_average(epochs, weights)
    return evoked_weighted_avg

#weights the epochs based on the variance of the data in each weight_epoch_lenght long epochs and returns new_epoch_lenght long weighted epochs
def variance_weighted_epochs(epochs, weight_epoch_lenght = None, new_epoch_lenght = None):

    #if no epoch lenght given, keep the one you have
    if weight_epoch_lenght == None:
        weight_epoch_lenght = epochs.times[-1]
    if new_epoch_lenght == None:
        new_epoch_lenght = epochs.times[-1]

    raw_conc = concatenate_epochs_raw(epochs)
    events_weight = create_equally_spaced_events(raw_conc, weight_epoch_lenght)
    epochs_weight = mne.Epochs(raw_conc, events=events_weight, tmin=0, tmax=weight_epoch_lenght-1/epochs.info['sfreq'], baseline=None, preload = True, detrend = None, on_missing = "ignore")

    epochs_weight_data = epochs_weight.get_data()
    variances = np.var(epochs_weight_data, axis=2)
    weights = np.where(variances>0, variances.shape[0]*(1/variances)/np.sum(1/variances,axis = 0), 1) # Why multiply by shape???

    weights_repeated = np.zeros(epochs_weight_data.shape)
    for i, epoch_w in enumerate(weights):
        for j, ch_w in enumerate(epoch_w):
            weights_repeated[i][j] = np.repeat(ch_w, epochs_weight_data.shape[2])

    weighted_epochs = epochs_weight_data*weights_repeated
    weighted_epochs = mne.EpochsArray(weighted_epochs, epochs_weight.info)

    raw_conc_weighted = concatenate_epochs_raw(weighted_epochs)

    events_epochs = create_equally_spaced_events(raw_conc_weighted, new_epoch_lenght)
    epochs_final = mne.Epochs(raw_conc_weighted, events=events_epochs, tmin=0, tmax=new_epoch_lenght-1/epochs.info['sfreq'], baseline=None, preload = True, detrend = None, on_missing = "ignore")
    return epochs_final

#MatLab like fast fourier transform amplitude spectrum function
def fft(evoked):
    picks = [x for x in evoked.pick(picks = 'eeg').info["ch_names"] if x not in evoked.pick(picks = 'eeg').info["bads"]]
    data = evoked.get_data(picks = picks)
    info = evoked.copy().pick(picks = picks).info

    L = len(data[0])
    #data_ft = 2/L*np.abs(fftw.fft(data))*(10**9)
    data_ft = 2/L*np.abs(fftw.fft(data))
    data_freq = np.round(100*np.arange(0,L)/L*info['sfreq'])/100
    print("L= "+str(L))
    spec = mne.time_frequency.SpectrumArray(data_ft, info, data_freq)
    return spec

#MatLab like fast fourier transform power spectrum function
def fftpowerspec(evoked):
    #picks = [x for x in evoked.pick(picks = 'eeg').info["ch_names"] if x not in evoked.pick(picks = 'eeg').info["bads"]]
    picks = [x for x in evoked.pick(picks = 'eeg').info["ch_names"]]
    data = evoked.get_data(picks = picks)
    info = evoked.copy().pick(picks = picks).info
    bads = info['bads']
    info['bads'] = []

    L = len(data[0])
    data_ft = (2/L*np.abs(fftw.fft(data)))**2
    data_freq = np.round(100*np.arange(0,L)/L*info['sfreq'])/100
    spec = mne.time_frequency.SpectrumArray(data_ft, info, data_freq)
    spec.info['bads'] = bads
    return spec

#MatLab like Butterworth filter function
def MatLabLike_Butterworth_filter(raw, PassBand, N = 4, copy = False, picks = 'eeg'):
    l_freq = PassBand[0]
    h_freq = PassBand[1]
    if l_freq == None or h_freq == None:
        padlen=3*(N)
    else:
        padlen=3*(2*N)
    iir_params = dict(order=N, ftype='butter', output='sos', padtype='odd', padlen= padlen)
    if copy:
        return raw.copy().filter(l_freq = l_freq, h_freq=h_freq, picks = picks, method='iir', iir_params=iir_params)
    else:
        return raw.filter(l_freq = l_freq, h_freq=h_freq, picks = picks, method='iir', iir_params=iir_params)
    
#filtfilt compliant with apply_function
def datafirstfiltfilt(data, b, a, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None):
    return sp.signal.filtfilt(b, a, data, axis=-1, padtype=padtype, padlen=padlen, method=method, irlen=irlen)

#MatLab like Butterworth filter function (LEGACY VERSION: USE MatLabLike_Butterworth_filter INSTEAD)
def MatLabLike_Butterworth_filter_noinfo(raw, PassBand, N = 4, copy = False, picks = 'eeg'):
    data = raw.get_data(picks = 'eeg')
    PassBand = np.array(PassBand)
    b, a = sp.signal.iirfilter(N, PassBand, rp=None, rs=None, btype='band', analog=False, ftype='butter', output='ba', fs=raw.info['sfreq'])
    if copy:
        return raw.copy().apply_function(datafirstfiltfilt, b = b, a = a, padtype='odd', padlen=3*(max(len(b),len(a))-1), picks = picks)
    else:
        return raw.apply_function(datafirstfiltfilt, b = b, a = a, padtype='odd', padlen=3*(max(len(b),len(a))-1), picks = picks)

#returns the good channels of an object... apparently it is not a built in function in mne
def good_channels(obj):
    good_chs = [x for x in obj.copy().pick(picks = 'eeg').info["ch_names"] if x not in obj.copy().pick(picks = 'eeg').info["bads"]]
    return good_chs


#cross reference earEEG channels from a raw object. left_chs will be the reference for right_chs and viceversa. extra_chs will be added without being part of the rereferencing (useful for triggers and other non-earEEG channels). WARNING: having channel referenced to different things in an MNE object will cause problems if you rereference again!!!
def cross_reference_raw(raw, left_chs, right_chs, extra_chs = []):
    pick_left_and_extra = left_chs.copy()
    pick_left_and_extra.extend(extra_chs)
    left_cross_raw = raw.copy().set_eeg_reference(ref_channels=right_chs).pick(picks = pick_left_and_extra)
    right_cross_raw = raw.copy().set_eeg_reference(ref_channels=left_chs).pick(picks = right_chs)

    left_cross_raw.add_channels([right_cross_raw], force_update_info=True)

    return left_cross_raw


#in ear reference earEEG channels from a raw object. left_chs will be the reference for left_chs and viceversa. extra_chs will be added without being part of the rereferencing (useful for triggers and other non-earEEG channels). WARNING: having channel referenced to different things in an MNE object will cause problems if you rereference again!!!
def inear_reference_raw(raw, left_chs, right_chs, extra_chs = []):
    pick_left_and_extra = left_chs.copy()
    pick_left_and_extra.extend(extra_chs)
    left_in_raw = raw.copy().set_eeg_reference(ref_channels=left_chs).pick(picks = pick_left_and_extra)
    right_in_raw = raw.copy().set_eeg_reference(ref_channels=right_chs).pick(picks = right_chs)

    left_in_raw.add_channels([right_in_raw], force_update_info=True)

    return left_in_raw

#reject completely flat channels from a raw object (checks if the difference between two consecutive time points is ever greater than a threshold, useful for saturated channels).
def reject_flat_chs(raw, threshold = 0.0001):
    data = raw.get_data()
    flat = []

    for ch in data:
        if np.abs(np.max(np.diff(ch))) < threshold:
            flat.append(True)
        else:
            flat.append(False)

    bad_flat = [x for i, x  in enumerate(raw.info['ch_names']) if flat[i]]
    raw.info['bads'].extend(bad_flat)

#reject non-significant ASSR channels from a raw object (checks if the modulation frequency point is significant in the power spectrum when referenced to ref_channel).
def reject_non_significant_ASSR_chs(raw, event, event_id = {}, ref_channel = 'Fz', TRIGGER_CLK = 18, THR_ERROR = -0.05, TRANS_ERROR = 0.1,
                                    AMfreq = 40, deltaf_low = 8, deltaf_high = 8):
    raw_copy = raw.copy()
    raw_copy.set_eeg_reference(ref_channels=[ref_channel])
    trig_data = np.logical_not((raw_copy.get_data(picks='Dig')[0]>0)).astype(int)

    decoder = SerialTriggerDecoder(trig_data, raw_copy.info['sfreq'], TRIGGER_CLK, THR_ERROR, TRANS_ERROR)
    decoded_trigs = decoder.decode()

    events = events_from_decoder(decoded_trigs)
    epochs = mne.Epochs(raw_copy, events=events, event_id=event_id, tmin=1/raw_copy.info['sfreq'], tmax=10, baseline=None, preload = True, detrend = None, on_missing = "ignore")
    evoked = epochs[event].average()

    fftspec = fft(evoked)
    significant = ASSR_Analysis(fftspec, AMfreq, deltaf_low, deltaf_high, significance = 0.01, tagging_frequency_tollerance = 0.1, power = False)['significant']

    bads_signal = [x for i, x  in enumerate(good_channels(raw_copy)) if not significant[i]]

    raw.info['bads'].extend(bads_signal)
    print('Rejected channels due to bad singal: ', bads_signal)
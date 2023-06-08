"""
Created April 2023
Last Edited on Mon May 22 2023
@author: Abra Geiger abrageiger7

Time and Frequency Varying Calculations of Fit Parameters
"""

#imports
import numpy as np
import matplotlib.pyplot as plt
import fit_functions as fittin
import convolved_pbfs as conv
import intrinsic_pbfs as intrins
import math

#import the parameter bank for reference, comparing, and plotting
convolved_profiles = conv.convolved_profiles
widths = conv.widths
gauss_widths = conv.widths_gaussian
betaselect = conv.betaselect
time = conv.time

#import data
data = np.load("J1903_data.npy")
freq = np.load("J1903_freqs.npy")
mjds = np.load("J1903_mjds.npy")
chan = np.load("J1903_numchan.npy")
dur = np.load("J1903_dur.npy")

#TO DO: find number of subintegrations, find best gauss width, find best beta
#eventually compare to decaying exponential, and intrinsic pulse shape convolution

#TO DO: Since fixed gaussian width units and chisquared, replot; fix in
#previous code difference between beta power law index and beta for pbfs

#Below are various calculations of fit parameters using functions from fit_functions.py

#=============================================================================
# Fitting with Beta PBFs and Decaying Exponential with new Class
    # Setting gwidth to index 4
    # Setting beta to index 11
# =============================================================================

mjd_list = []
freq_list = []
dur_list = []
subavg_chan_list = []


pbf_width_listb = []
low_chi_listb = []
tau_listb = []
tau_low_listb = []
tau_high_listb = []
gauss_width_listb = []
fse_listb = []

pbf_width_liste = []
low_chi_liste = []
tau_liste = []
tau_low_liste = []
tau_high_liste = []
gauss_width_liste = []
fse_liste = []


for i in range(56):
    sub_int = True
    ii = 0
    while sub_int == True:
        num_chan0 = int(chan[i])
        data0 = data[i][:num_chan0]
        freq0 = freq[i][:num_chan0]
        p = fittin.Profile(mjds[i], data0, freq0, ii, dur[i])]
        if ii == 0:
            subavg_chan_list.append(p.num_sub)

        dur_list.append(dur[i])
        mjd_list.append(mjds[i])
        freq_list.append(p.freq_suba)

        datab = p.fit(beta_ind = 11, gwidth_ind = 4)
        gauss_width_listb.append(datab[5])
        pbf_width_listb.append(datab[6])
        low_chi_listb.append(datab[0])
        tau_listb.append(datab[1])
        tau_low_listb.append(datab[2])
        tau_high_listb.append(datab[3])
        fse_listb.append(datab[4])


        datae = p.fit(gwidth_ind = 4, dec_exp = True)
        gauss_width_liste.append(datae[5])
        pbf_width_liste.append(datae[6])
        low_chi_liste.append(datae[0])
        tau_liste.append(datae[1])
        tau_low_liste.append(datae[2])
        tau_high_liste.append(datae[3])
        fse_liste.append(datae[4])


        ii += 1
        if ii > p.num_sub - 1:
            sub_int = False

setg4setb11_data = np.array([mjd_list, freq_list, pbf_width_listb, low_chi_listb, tau_listb, tau_low_listb, tau_high_listb, gauss_width_listb])

np.save('setg4setb11_data', setg4setb11_data)

setg4dece_data = np.array([mjd_list, freq_list, pbf_width_liste, low_chi_liste, tau_liste, tau_low_liste, tau_high_liste, gauss_width_liste])

np.save('setg4dece_data', setg4dece_data)

np.save('J1903_subavgnumchan', subavg_chan_list)

#=============================================================================
# Test of new class and errors on tau
# =============================================================================

# num_chan0 = int(chan[40])
# data0 = data[40][:num_chan0]
# freq0 = freq[40][:num_chan0]
#
# p1 = fittin.Profile(mjds[40],data0,freq0,0)
#
# data_testb = p1.fit(beta_ind = 11, gwidth_ind = 4)
# data_teste = p1.fit(gwidth_ind = 4, dec_exp = True)
#
# np.save('data_testb_h', data_testb)
# np.save('data_teste_h', data_teste)
#
# num_chan0 = int(chan[40])
# data0 = data[40][:num_chan0]
# freq0 = freq[40][:num_chan0]
#
# p1 = fittin.Profile(mjds[40],data0,freq0,0)
#
# data_testb = p1.fit(beta_ind = 11, gwidth_ind = 4)
# data_teste = p1.fit(gwidth_ind = 4, dec_exp = True)
#
# np.save('data_testb_h', data_testb)
# np.save('data_teste_h', data_teste)

#=============================================================================
# Fitting with Decaying Exponential and Gaussian Width
    # Setting gwidth to index 4
# =============================================================================

# mjd_listeg = []
# freq_listeg = []
# pbf_width_listeg = []
# low_chi_listeg = []
# tau_listeg = []
# gauss_width_listeg = []
#
# gwidth_index = 4
#
# for i in range(56):
#     sub_int = True
#     ii = 0
#     while sub_int == True:
#         num_chan0 = int(chan[i])
#         data0 = data[i][:num_chan0]
#         freq0 = freq[i][:num_chan0]
#         dataer = fittin.fit_dec_setgwidth_exp(mjds[i], data0, freq0, ii, gwidth_index)
#         mjd_listeg.append(mjds[i])
#         freq_listeg.append(dataer[4])
#         gauss_width_listeg.append(dataer[2])
#         pbf_width_listeg.append(dataer[3])
#         low_chi_listeg.append(dataer[0])
#         tau_listeg.append(dataer[1])
#         ii += 1
#         if ii > dataer[5] - 1:
#             sub_int = False
#
# setgdece_arrayyay = np.array([mjd_listeg, freq_listeg, gauss_width_listeg, pbf_width_listeg, low_chi_listeg, tau_listeg])
#
# np.save('setgdece_arrayyay', setgdece_arrayyay)


#=============================================================================
# Fitting with Constant Beta and Gaussian Width
    # Setting gwidth to index 4
    # Setting beta to index 11
# =============================================================================

# mjd_listgb = []
# beta_listgb = []
# freq_listgb = []
# pbf_width_listgb = []
# low_chi_listgb = []
# tau_listgb = []
# gauss_width_listgb = []
# subavg_chan_listgb = []
#
# gwidth_index = 4
# beta_index = 11
#
# for i in range(56):
#     sub_int = True
#     ii = 0
#     while sub_int == True:
#         num_chan0 = int(chan[i])
#         data0 = data[i][:num_chan0]
#         freq0 = freq[i][:num_chan0]
#         dataer = fittin.fit_cons_beta_gauss_profile(mjds[i], data0, freq0, ii, beta_index, gwidth_index)
#         mjd_listgb.append(mjds[i])
#         freq_listgb.append(dataer[4])
#         beta_listgb.append(betaselect[beta_index])
#         gauss_width_listgb.append(dataer[2])
#         pbf_width_listgb.append(dataer[3])
#         low_chi_listgb.append(dataer[0])
#         tau_listgb.append(dataer[1])
#         subavg_chan_listgb.append(dataer[5])
#         ii += 1
#         if ii > dataer[5] - 1:
#             sub_int = False
#
# setgsetb_arrayyay = np.array([mjd_listgb, beta_listgb, freq_listgb, gauss_width_listgb, pbf_width_listgb, low_chi_listgb, tau_listgb, subavg_chan_listgb])
#
# np.save('setgsetb_arrayyay', setgsetb_arrayyay)

#=============================================================================
# Fitting Instrinsic Pulse
# =============================================================================

# fittin.fit_cons_beta_ipfd(mjds[0], data0, freq0, 0, 11)

#=============================================================================
# Setting Constant Gaussian Width
#   Setting all to gwidths index 4
# =============================================================================
# it seems that 50 was the best across most frequencies (FWHM in terms of phase bins)
#print(50/(2.0*math.sqrt(2*math.log(2))))
#print(gauss_widths * (2.0*math.sqrt(2*math.log(2))))
#rint(gauss_widths)

# =============================================================================
# mjd_listg = []
# beta_listg = []
# freq_listg = []
# pbf_width_listg = []
# low_chi_listg = []
# tau_listg = []
# gauss_width_listg = []
#
# gwidth_index = 4
#
# for i in range(5):
#     for ii in range(12):
#         num_chan0 = int(chan[i*10])
#         data0 = data[i*10][:num_chan0]
#         freq0 = freq[i*10][:num_chan0]
#         dataer = fittin.fit_all_profile_set_gwidth(mjds[i*10], data0, freq0, ii, gwidth_index)
#         mjd_listg.append(mjds[i*10])
#         beta_listg.append(dataer[4])
#         freq_listg.append(dataer[5])
#         gauss_width_listg.append(dataer[2])
#         pbf_width_listg.append(dataer[3])
#         low_chi_listg.append(dataer[0])
#         tau_listg.append(dataer[1])
#
#
# setg_arrayyay = np.array([mjd_listg, beta_listg, freq_listg, gauss_width_listg, pbf_width_listg, low_chi_listg, tau_listg])
#
# np.save('setg_betadatayay', setg_arrayyay)
# =============================================================================

# =============================================================================

# num_chan0 = int(chan[40])
# data0 = data[40][:num_chan0]
# freq0 = freq[40][:num_chan0]

# low_chi, tau_fin, gauss_width_fin, pbf_width_fin, beta_fin, freqs_care = \
#    fittin.fit_all_profile(mjds[40], data0, freq0, 10)
# np.save('Varying_Beta'+str(mjds[40])[:5]+'lowfreq', \
#        np.array([low_chi, tau_fin, gauss_width_fin, pbf_width_fin, \
#                  beta_fin, freqs_care]))

# =============================================================================

# num_chan0 = int(chan[0])
# data0 = data[0][:num_chan0]
# freq0 = freq[0][:num_chan0]

# fittin.fit_all_profile(mjds[0], data0, freq0, 11) # for cons beta 11, plot_conv=True

# low_chi, tau_fin, gauss_width_fin, pbf_width_fin, beta_fin, freqs_care = \
#     fittin.fit_all_profile(mjds[0], data0, freq0, 0)
# np.save('Varying_Beta'+str(mjds[0])[:5]+'highfreq', \
#         np.array([low_chi, tau_fin, gauss_width_fin, pbf_width_fin, \
#                   beta_fin, freqs_care]))

# =============================================================================

# num_chan0 = int(chan[25])
# data0 = data[25][:num_chan0]
# freq0 = freq[25][:num_chan0]
#
# low_chi, tau_fin, gauss_width_fin, pbf_width_fin, beta_fin, freqs_care = \
#     fittin.fit_all_profile(mjds[25], data0, freq0, 0)
# np.save('Varying_Beta'+str(mjds[25])[:5]+'highfreq', \
#         np.array([low_chi, tau_fin, gauss_width_fin, pbf_width_fin, \
#                   beta_fin, freqs_care]))

# =============================================================================

# fittin.fit_dec_exp(mjds[0], data0, freq0, 0)
# fittin.fit_cons_beta_profile(mjds[0], data0, freq0, 0, 11)
# fittin.fit_cons_beta_ipfd(mjds[0], data0, freq0, 0, 11)



#=============================================================================
#comparing over frequency
# =============================================================================
# freqc = np.zeros(10)
# chisa_pf = np.zeros(10)
# tausa_pf = np.zeros(10)
# gaussa_pf = np.zeros(10)
# pbfsa_pf = np.zeros(10)
# beta_set = 11

# for i in range(10):
#     low_chi, tau_fin, gaussian_width_fin, pbf_width_fin, freqs_care = \
#         fittin.fit_cons_beta_profile(mjds[0], data0, freq0, i, beta_set)
#     chisa_pf[i] = low_chi
#     tausa_pf[i] = tau_fin
#     gaussa_pf[i] = gaussian_width_fin
#     pbfsa_pf[i] = pbf_width_fin
#     freqc[i] = freqs_care

# np.save('Beta=4_'+str(mjds[0])[:5]+'_varyingfreq', np.array([freqc, chisa_pf, tausa_pf, gaussa_pf, pbfsa_pf]))

# chisa_ef = np.zeros(10)
# tausa_ef = np.zeros(10)
# gaussa_ef = np.zeros(10)
# pbfsa_ef = np.zeros(10)

# for i in range(10):
#     low_chi, tau_fin, gaussian_width_fin, pbf_width_fin, freqs_care = \
#         fittin.fit_dec_exp(mjds[0], data0, freq0, i)
#     chisa_ef[i] = low_chi
#     tausa_ef[i] = tau_fin
#     gaussa_ef[i] = gaussian_width_fin
#     pbfsa_ef[i] = pbf_width_fin

# np.save('Exp_'+str(mjds[0])[:5]+'_varyingfreq', np.array([freqc, chisa_ef, tausa_ef, gaussa_ef, pbfsa_ef]))

#============================================================================
#comparing over mjd
# =============================================================================
# mjdc = np.zeros(10)
# chisa_pm = np.zeros(10)
# tausa_pm = np.zeros(10)
# gaussa_pm = np.zeros(10)
# pbfsa_pm = np.zeros(10)
# beta_set = 11

# index = 0
# for i in np.arange(0,50,5):
#     num_chan0 = int(chan[i])
#     data0 = data[i][:num_chan0]
#     freq0 = freq[i][:num_chan0]
#     low_chi, tau_fin, gaussian_width_fin, pbf_width_fin, freqs_care = \
#         fittin.fit_cons_beta_profile(mjds[i], data0, freq0, 0, beta_set)
#     mjdc[index] = mjds[i]
#     chisa_pm[index] = low_chi
#     tausa_pm[index] = tau_fin
#     gaussa_pm[index] = gaussian_width_fin
#     pbfsa_pm[index] = pbf_width_fin
#     index += 1

# np.save('Beta=4_varyingmjd_highfreq', np.array([mjdc, chisa_pm, tausa_pm, gaussa_pm, pbfsa_pm]))

# chisa_em = np.zeros(10)
# tausa_em = np.zeros(10)
# gaussa_em = np.zeros(10)
# pbfsa_em = np.zeros(10)

# index = 0
# for i in np.arange(0,50,5):
#     num_chan0 = int(chan[i])
#     data0 = data[i][:num_chan0]
#     freq0 = freq[i][:num_chan0]
#     low_chi, tau_fin, gaussian_width_fin, pbf_width_fin, freqs_care = \
#         fittin.fit_dec_exp(mjds[i], data0, freq0, 0)
#     chisa_em[index] = low_chi
#     mjdc[index] = mjds[i]
#     tausa_em[index] = tau_fin
#     gaussa_em[index] = gaussian_width_fin
#     pbfsa_em[index] = pbf_width_fin
#     index += 1

# np.save('Exp_varyingmjd_highfreq', np.array([mjdc, chisa_em, tausa_em, gaussa_em, pbfsa_em]))

#=============================================================================
# Comparing Beta over Frequency and MJD
# =============================================================================
# mjd_list = []
# beta_list = []
# freq_list = []
# gauss_width_list = []
# pbf_width_list = []
# low_chi_list = []
# tau_list = []

# for i in range(5):
#     for ii in range(12):
#         num_chan0 = int(chan[i*10])
#         data0 = data[i*10][:num_chan0]
#         freq0 = freq[i*10][:num_chan0]
#         dataer = fittin.fit_all_profile(mjds[i*10], data0, freq0, ii)
#         mjd_list.append(mjds[i*10])
#         beta_list.append(dataer[4])
#         freq_list.append(dataer[5])
#         gauss_width_list.append(dataer[2])
#         pbf_width_list.append(dataer[3])
#         low_chi_list.append(dataer[0])
#         tau_list.append(dataer[1])

# arrayyay = np.array([mjd_list, beta_list, freq_list, gauss_width_list, pbf_width_list, low_chi_list, tau_list])

# np.save('betadatayay', arrayyay)

#=============================================================================

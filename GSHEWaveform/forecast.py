#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Miguel Zumalacarregui, Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Created on Fri Apr 28 17:05:27 2023

@author: miguel

Uses https://github.com/hsinyuc/distancetool, cite Chen et al (2017)
(arXiv:1709.08079)
"""

import sys

import numpy as np
import pandas as pd
import pycbc as pycbc
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from pycbc.filter.matchedfilter import optimized_match
from scipy.integrate import simps
from scipy.interpolate import griddata, interp1d

sys.path.insert(0,'/home/miguel/code/utils/distancetool/codes/')
import find_horizon_range_de as gwhor

sensitivities_dir = '../../glow/sensitivities/'

#Load the data (computed in )
data_dict = np.load("../data/Ups_obs_mu_min.npy", allow_pickle=True).item()
betalims = data_dict["betalims"]
mu_min_arr = data_dict["mu_min_arr"]
ups_obs_arr = data_dict["ups_obs_arr"]


def d_ups_obs_dmu(ups_obs_arr, mu_min_arr):
    r"""
    Compute the derivative of :math:`Upsilon_{\rm obs}` with respect to
    magnification.

    Parameters
    ----------
    ups_obs_arr : array
        Array of :math:`Upsilon_{\rm obs}` values.
    mu_min_arr : array
        Array of :math:`\mu_{\rm min}`, the magnification limit values.

    Returns
    -------
    dups_obs_dmu : array
    """
    dups_obs_dmu = np.zeros_like(ups_obs_arr)
    for i in range(0, len(mu_min_arr) - 1):
        dups_obs_dmu[i, :] = ups_obs_arr[i + 1] - ups_obs_arr[i]
        dups_obs_dmu[i, :] /= mu_min_arr[i + 1] - mu_min_arr[i]
    return dups_obs_dmu

dups_obs_dmu = d_ups_obs_dmu(ups_obs_arr, mu_min_arr)

psd_ligo_read=pd.read_csv(sensitivities_dir+"aplus.txt", sep=" ", header=None ,index_col=None)
psd_ligo_arr=np.transpose(psd_ligo_read.to_numpy(dtype=float))
psd2=interp1d(psd_ligo_arr[0],psd_ligo_arr[1],kind='linear', fill_value='extrapolate')

psd_et_read = pd.read_csv(sensitivities_dir+"et.txt", sep=" ", header=None ,index_col=None)
psd_et_arr=np.transpose(psd_et_read.to_numpy(dtype=float))
psd_et2=interp1d(psd_et_arr[0],psd_et_arr[1],kind='linear', fill_value='extrapolate')


def psd_ligo(f):
    '''Power spectral density from https://dcc.ligo.org/LIGO-T1500293/public'''
    return float(psd2(f)**2)

def psd_et(f):
    '''Power spectral density for ET'''
    return float(psd_et2(f)**2)

psd_ligo=np.vectorize(psd_ligo)
psd_et = np.vectorize(psd_et)

aligo_file = gwhor.base_dir+'/data/aLIGO/Advanced_LIGO_Design.txt'
CE1_file = gwhor.base_dir+'/data/CE1_strain.txt'
CE2_file = gwhor.base_dir+'/data/CE1_strain.txt'


def psd_from_file(asdfile,fmin=0):

    input_freq,strain=np.loadtxt(asdfile,unpack=True,usecols=[0,1])
    if fmin==0:
        fmin = np.min(input_freq)
    minimum_freq=np.maximum(min(input_freq),fmin)
    maximum_freq=np.minimum(max(input_freq),5000.)
    interpolate_psd = interp1d(input_freq, strain**2,kind='linear', fill_value='extrapolate')
    return interpolate_psd

CE_psd = psd_from_file(CE1_file)


def mismatch_gshe(M_bbh, q=1,M_fid = 1e4,bt_fid = 0.1, z=0.3,fmin=10, fref=10, fmax=5e3, df=1,n_sample=5, psd_fun = psd_ligo,approx=gwhor.ls.IMRPhenomD):
    '''compute the fiducial mismatch and SNR'''
    
    t_obs= 1/12/30/24/60/2
    m1 = M_bbh/(1+q)
    m2 = q*m1
#     fs, hx, hp = waveform(M_bbh, q=q, z=zS_waveform, t_obs=t_obs, f_max_wrt_isco=f_max_wrt_isco, n_sample=n_sample)

    hp,hx,fs= gwhor.get_htildas((1.+z)*m1,(1.+z)*m2 , gwhor.de.luminosity_distance_de(z,**gwhor.cosmo),fmin=fmin,fref=fref,df=df,approx=approx)
    fsel=np.logical_and(fs>fmin,fs<fmax)
#     psd_interp = gwhor.interpolate_psd(fs[fsel]) 	

    df = fs[1]-fs[0]
    hp = pycbc.types.frequencyseries.FrequencySeries(hp,df)
    
    psd = pycbc.types.frequencyseries.FrequencySeries(psd_fun(fs), df)
    snr0 = gwhor.compute_horizonSNR(hp,psd[fsel],fsel,df)
#     snr0 = pycbc.filter.matched_filter(hp, hp, psd=psd, low_frequency_cutoff=fmin)
    
    f0_fid = bt_fid/(M_fid*1e-5) #in Hz

    F_gshe = np.exp(2*np.pi*1j*f0_fid/(fs))
    F_gshe[F_gshe==np.nan]=1

    
    h_gshe = pycbc.types.frequencyseries.FrequencySeries(F_gshe*hp, df)

    mismatch_fid = 1-optimized_match(hp, h_gshe, psd = psd,low_frequency_cutoff=fmin)[0]
    
    return [mismatch_fid, np.abs(snr0)]


def beta_min(M_bbh, z=0.3, q=1,M_fid = 1e4,bt_fid = 0.1,fmin=10, fref=10, fmax=5e3, df=1,n_sample=5, psd_fun = psd_ligo,approx=gwhor.ls.IMRPhenomD):
    '''convenient wrapper to compute the minimum beta, using the bt^2 dependence of the misamatch'''
    M,SNR0 = mismatch_gshe(M_bbh, q,M_fid,bt_fid, z,fmin, fref, fmax, df,n_sample, psd_fun,approx)
    return bt_fid/np.sqrt(M)/SNR0

beta_min = np.vectorize(beta_min)


def prob_GSHE_interp(bt,j_mu_min):
    '''
    interpolate dUps_obs/dmu
    bt: beta_min values
    j_mu_min: column for mu_min
    returns negative because mu_min is the lower limit in the cumulatime dsitibution
    '''
    P = griddata(betalims,dups_obs_dmu[j_mu_min,:],bt)
    P[bt<betalims.min()] = dups_obs_dmu[j_mu_min,0]
    P[bt>betalims.max()] = 0
    return -P


def GSHE_rates(M_tot,psd_fun,psd_file, snr_th=8, M_fid=1e4,zs=np.geomspace(0.01,10,30), R0 = 30, return_arrays=False, normalize=True):
    ''' computes GSHE rates, effective volume and derived quantities
    '''
    
    results = {}
    
    #1D quantities
    mus = mu_min_arr

    btmin0 = beta_min(M_tot,zs,psd_fun=psd_fun,M_fid = M_fid)
    snr_opt = SNR_opt(zs,M_tot,psd_file)
    dVz = (cosmo.differential_comoving_volume(zs)/u.Gpc**3*4*np.pi*u.sr/(1+zs)).decompose()
    
    #generate 2D quantities
    p_det = prob_detect((snr_th/snr_opt)[:,None]/np.sqrt(mus))
    btmin = btmin0[:,None]/np.sqrt(mus)

    P_GSHE = np.array([prob_GSHE_interp(btmin[:,j],j) for j,mu in enumerate(mus)]).T
    #renormalize
    if normalize: 
        P_GSHE[P_GSHE>1] = 1
    dVzdz = dVz[:,None]/mus**0
    
    ##NOTE: PREVIOUSLY np.array(dVzdz*p_det*btmin*P_GSHE) WITH btmin?
    integrand = np.array(dVzdz*p_det*P_GSHE)
    GSHE_vol = simps(simps(dVzdz*p_det*P_GSHE, mus), zs)
    
    #just remove magnified events
    #need to include a more realistic magnification function
    N_tot = simps(simps(dVzdz*p_det*np.heaviside(1-mus,1), mus), zs)
    
    results['SNR_th'] = snr_th
    results['M_fid'] = M_fid
    results['psd_file'] = psd_file
    results['M_bbh'] = M_tot
    
    results['GSHE_volume [Gpc^3]'] = GSHE_vol
    results['R0'] = R0
    results['rate GSHE [1/yr]'] = GSHE_vol*R0
    results['rate detection [1/yr]'] = N_tot*R0
    if return_arrays:
        results['mu_vals'] = mus
        results['z_vals'] = zs
        results['integrand'] = integrand
        results['P_GSHE'] = P_GSHE
        results['p_det'] = p_det
        results['dVzdz'] = dVzdz

    return results


z_grid = np.geomspace(1e-3, 500,1000)
DLMpc_grid = (cosmo.luminosity_distance(z_grid)/u.Mpc).decompose()

def z_real_gw(z_obs,mu,cosmo=cosmo):
    '''
    real redshift for a GW observation
    z_obs: observed redshift
    mu: magnification
    '''
    DL_real = np.sqrt(mu)*cosmo.luminosity_distance(z_obs)/u.Mpc
    
    z_real = griddata(DLMpc_grid,z_grid, DL_real)
    return z_real
    
z_real_gw = np.vectorize(z_real_gw)  


#from Chen, Holz+
#sampled universal antenna power pattern for code sped up
w_sample,P_sample=np.genfromtxt(gwhor.base_dir+"/data/Pw_single.dat",unpack=True)
P_detect=interp1d(w_sample, P_sample,bounds_error=False,fill_value=0.0)
dP_sample = np.gradient(P_sample,w_sample)
dP_dx = interp1d(w_sample,dP_sample,bounds_error=False,fill_value=0.0)

def prob_detect(w):
    '''
    detection probability
    w = snr_th/snr_opt
    '''
    P = griddata(w_sample,P_sample,w)
    P[w>w_sample.max()] = 0
    P[w<w_sample.min()] = 1
    return P


def guess_z_horizon(m1,m2, asdfile, snr_th = 8,fmin=10.,fref=10.,df=1.,maximum_freq = 1e3,approx = gwhor.ls.IMRPhenomD):
    
    input_freq,strain=np.loadtxt(asdfile,unpack=True,usecols=[0,1])
    print(min(input_freq))
    minimum_freq=np.maximum(min(input_freq),fmin)
    maximum_freq=np.minimum(max(input_freq),5000.)
    interpolate_psd = interp1d(input_freq, strain**2)
    
    #initial guess of horizon redshift and luminosity distance
    z0=1.0
    input_dist=gwhor.de.luminosity_distance_de(z0,**gwhor.cosmo)	
    hplus_tilda,hcross_tilda,freqs= gwhor.get_htildas((1.+z0)*m1,(1.+z0)*m2 ,input_dist,iota=0.,fmin=fmin,fref=fref,df=df,approx=approx)
    fsel=np.logical_and(freqs>minimum_freq,freqs<maximum_freq)
    psd_interp = interpolate_psd(freqs[fsel]) 	
    input_snr=gwhor.compute_horizonSNR(hplus_tilda,psd_interp,fsel,df)

    input_redshift=z0; guess_snr=0; njump=0
    #evaluate the horizon recursively
    while abs(guess_snr-snr_th)>snr_th*0.001 and njump<10: #require the error within 0.1%
        try:
            guess_redshift,guess_dist=gwhor.horizon_dist_eval(input_dist,input_snr,input_redshift) #horizon guess based on the old SNR		
            hplus_tilda,hcross_tilda,freqs= gwhor.get_htildas((1.+guess_redshift)*m1,(1.+guess_redshift)*m2 ,guess_dist,iota=0.,fmin=fmin,fref=fref,df=df,approx=approx)
        except:
            njump=10
            print("Will try interpolation.")		
        fsel=np.logical_and(freqs>minimum_freq,freqs<maximum_freq)
        psd_interp = interpolate_psd(freqs[fsel]) 		
        guess_snr=gwhor.compute_horizonSNR(hplus_tilda,psd_interp,fsel,df) #calculate the new SNR

        input_snr=guess_snr
        input_redshift=guess_redshift
        input_dist=guess_dist
        njump+=1
        #print(njump,guess_snr,guess_redshift)
    horizon_redshift=guess_redshift



	#at high redshift the recursive jumps lead to too big a jump for each step, and the recursive loop converge slowly.
    #so I interpolate the z-SNR curve directly.	
    if njump>=10:
        print("Recursive search for the horizon failed. Interpolation instead.")
        try:
            interp_z = np.linspace(0.001,120,1000)
            interp_snr = np.zeros(interp_z.size)
            for i in range(0, interp_z.size):
                hplus_tilda,hcross_tilda,freqs= gwhor.get_htildas((1.+interp_z[i])*m1,(1.+interp_z[i])*m2 ,gwhor.de.luminosity_distance_de(interp_z[i],**gwhor.cosmo),fmin=fmin,fref=fref,df=df,approx=approx)
                fsel=np.logical_and(freqs>minimum_freq,freqs<maximum_freq)
                psd_interp = interpolate_psd(freqs[fsel]) 	

                interp_snr[i]=gwhor.compute_horizonSNR(hplus_tilda,psd_interp,fsel,df)	
            interpolate_snr = interp1d(interp_snr[::-1],interp_z[::-1])
            horizon_redshift= interpolate_snr(snr_th)	
        except RuntimeError: #If the sources lie outside the given interpolating redshift the sources can not be observe, so I cut down the interpolation range.
            print("some of the SNR at the interpolated redshifts cannot be calculated.")
            interpolate_snr = interp1d(interp_snr[::-1],interp_z[::-1])
            horizon_redshift= interpolate_snr(snr_th)	
        except ValueError:	#horizon outside the interpolated redshifts. Can potentially modify the interpolation range, but we basically can not observe the type of source or the source has to be catastrophically close.
            print("Horizon further than z=120 or less than z=0.001")
            return	
    #horizon_redshift=30.
    hplus_tilda,hcross_tilda,freqs= gwhor.get_htildas((1.+horizon_redshift)*m1,(1.+horizon_redshift)*m2 ,gwhor.de.luminosity_distance_de(horizon_redshift,**gwhor.cosmo),fmin=fmin,fref=fref,df=df,approx=approx)
    fsel=np.logical_and(freqs>minimum_freq,freqs<maximum_freq)
    psd_interp = interpolate_psd(freqs[fsel]) 	

    print("SNR ",gwhor.compute_horizonSNR(hplus_tilda,psd_interp,fsel,df), ' at z=',horizon_redshift)
    return horizon_redshift 


def SNR_opt(z,M,asdfile,q=1,fmin=10.,fref=10.,df=1.,maximum_freq = 5e3,approx = gwhor.ls.IMRPhenomD):
    """
    optimal SNR(z)
    accounts for source's redshift
    NOTE: might have problems if redshift is too high
    """
    m1 = M*(1-q/2)
    m2 = m1/q
    snr_th = 8
    input_freq,strain=np.loadtxt(asdfile,unpack=True,usecols=[0,1])
    minimum_freq=np.maximum(min(input_freq),fmin)
    maximum_freq=np.minimum(max(input_freq),5000.)
    interpolate_psd = interp1d(input_freq, strain**2)
    optsnr_z = np.zeros_like(z)

    for i in range(0,np.size(z)):	
        hplus_tilda,hcross_tilda,freqs = gwhor.get_htildas((1.+z[i])*m1,(1.+z[i])*m2, gwhor.de.luminosity_distance_de(z[i],**gwhor.cosmo) ,fmin=fmin,fref=fref,df=df,approx=approx)
        fsel = np.logical_and(freqs>minimum_freq,freqs<maximum_freq)
        psd_interp = interpolate_psd(freqs[fsel])  
        optsnr_z[i]= gwhor.compute_horizonSNR(hplus_tilda,psd_interp,fsel,df)
    return optsnr_z

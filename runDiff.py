# --- Common libraries
import os
import numpy as np
import pandas as pd
import scipy as sp
from scipy import optimize
# --- Local libraries
import weio
from MiniBEM import MiniBEM, FASTFile2MiniBEM, runParametricBEM_FAST
from wake_model import get_u_turb
# from helpers import read_excel
from helpers import *

if __name__=="__main__":
    """ setup BEM """
    OutDir                 = '_dataTmp'
    MainFASTFile           = '../data/BAR-WT/RotorSE_FAST_BAR_005a.fst'
    OperatingConditionFile = '../data/BAR_Rigid_Performances_FST.csv'

    #  --- Reading turbine operating conditions, Pitch, RPM, WS  (From FAST)
    nB, cone, r, chord, twist, polars, rho, KinVisc = FASTFile2MiniBEM(MainFASTFile)
    df=weio.read(OperatingConditionFile).toDataFrame()
    Pitch = df['BldPitch_[deg]'].values
    Omega = df['RotSpeed_[rpm]'].values
    WS    = df['WS_[m/s]'].values
    dfOut = runParametricBEM_FAST(MainFASTFile, Pitch, Omega, WS)
    # 5 corresponds to Uinf = 8 m/s
    V0 = WS[5]  # [m/s]
    RPM = Omega[5]  # [rpm]
    pitch = Pitch[5]  # [deg]
    u_turb = 0  # [m/s]
    """ end setup BEM """

    run = "S3G1"
    x0 = [-0.5, 0, 0.5] #up, down, mid
    nacelle_L = 20  # m
    ab = np.array([1/2, 1/8])
    Uinf = 8  # m/s
    Cd = 0.2
    wake = 0
    for i in range(0,2):
        CFD_path = "/Users/banderso2/Documents/Office/BAR/2019-nacelle-blockage/data/Fluent/"+run+"x"+str(x0[i])+".csv"
        diff(ab, nacelle_L, x0[i], Uinf, Cd, wake, CFD_path, RPM, pitch, V0, nB, cone, r, chord, twist, polars, rho)
        BEM = MiniBEM(RPM, pitch, V0, xdot=0, u_turb,
                      nB, cone, r, chord, twist, polars,
                      rho=rho, KinVisc=KinVisc, bTIDrag=False, bAIDrag=True,
                      a_init=None, ap_init=None)


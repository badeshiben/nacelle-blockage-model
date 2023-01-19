""" 
Performs simple BEM simulations of the BAR turbines for different operating conditions.
"""
# --- Common libraries 
import os
import numpy as np
import pandas as pd
import scipy as sp
from scipy import optimize
# --- Local libraries
import weio
from MiniBEM import MiniBEM, FASTFile2MiniBEM
from wake_model import get_u_turb
# from helpers import read_excel
from helpers import *


def runParametricBEM_FAST(MainFASTFile, Pitch, Omega, WS):
    """ 
    Performs BEM simulations for different Pitch, RPM and Wind Speed.
    The wind turbine is initialized using a FAST input file (.fst)
    """
    # -- Extracting information from a FAST main file and sub files
    nB,cone,r,chord,twist,polars,rho,KinVisc = FASTFile2MiniBEM(MainFASTFile) 
    BladeData=np.column_stack((r,chord,twist))

    """ get parameters from FAST file"""
    if not os.path.exists(OutDir):
        os.makedirs(OutDir)
    # run multiple WS
    # for i in np.arange(len(df)):
    #     V0    = WS[i]    # [m/s]
    #     RPM    = Omega[i] # [rpm]
    #     pitch  = Pitch[i] # [deg]
    #     xdot   = 0        # [m/s]
    #     u_turb = 0        # [m/s]
    # run 1 WS
    if 1:
        V0 = WS[5]  # [m/s]
        RPM = Omega[5]  # [rpm]
        pitch = Pitch[5]  # [deg]
        xdot = 0  # [m/s]
        u_turb = 0  # [m/s]

        """################BEGIN USER INPUT##################"""
        a0  = None # Initial guess for axial induction, to speed up BEM
        ap0 = None # Initial guess for tangential induction, to speed up BEM
        dfOut=None
        optimization = False  #bool flag to run optimization or not

        type = 'rotor'  # solve along rotor or grid
        plot = 0  # 0=no plot; 1=display plot; 2=save plot
        export = True  # flag to export rotor results to csv file
        nacelle_L = 20  # 14.5988  # m

        blockage = 'none'  # Nacelle blockage flag. 'none', 'EM', 'CFD'
        orientation = 'j'#"downwind"
        root_airfoils = "polar7"
        suite = "Suite3"
        run = "geometry3"
        run_name = suite+run

        if orientation == "downwind":
            x0 = 0.5  # location of rotor root along nacelle [nacelle lengths]. x0=0 is at the nacelle midplane
        else:
            x0 = 0

        excel_name = "Nacelle_blockage_{}_results_{}_{}.xlsx".format(blockage, orientation, root_airfoils)
        """CFD"""
        z0 = 0  # -0.25 for first rectangle #nacelle z-midplane location
        """EM"""
        cone = 0  # cone angle [rad]
        Cd = 0.2  # nacelle drag coefficient
        a = 0.5  # half nacelle major axis [nacelle lengths]
        b = 0.125  # half nacelle minor axis [nacelle lengths]
        wake = False  # wake model flag
        """##################END USER INPUT######################"""

        """ run BEM simulations"""
        if optimization:
            res = optimize.minimize(fun=diff, x0=np.array([0.8, 0.25]), args=(nacelle_L), bounds=((0.5, 1.5), (0.15, 0.49)),
                           options={'disp': True})

        if blockage=='EM':
            #u_turb = get_u_turb(x0=12, r=r, cone=cone, Uinf=V0, Cd=1, a=10, b=5, plot=2, export=1)        # [m/s] TODO add radial wake induction profile to u_turb
            # r=np.linspace(0, 105, 1000)
            u_turb = get_u_turb(r, x0, cone, V0, Cd, a, b, wake, type, plot, export)
        elif blockage =='CFD':
            # CFD = pd.read_excel("../data/y0/nacelle_blockage_LES.xls", sheet_name=str(x0)+"_comparison_data")
            CFD = read_excel("../data/CFD/{}/{}/z_{:01.1f}.csv".format(suite, run, x0+0.5), x0, nacelle_L, z0, Uinf=V0)
            r_CFD = CFD.loc[:, "y"].to_numpy()*nacelle_L
            r_CFD = np.insert(r_CFD, 0, 0)
            u_turb_CFD = CFD.loc[:, "u_turb"].to_numpy()
            u_turb_CFD = np.insert(u_turb_CFD, 0, -1)
            f = sp.interpolate.interp1d(r_CFD, u_turb_CFD)
            u_turb = f(r)
        if blockage != 'none':
            u_turb_avg_percent = np.mean(u_turb) * 100
            u_turb_max_percent = max(u_turb) * 100
            u_turb_min_percent = min(u_turb) * 100

        if type == 'rotor':
            BEM=MiniBEM(RPM,pitch,V0,xdot,u_turb,
                        nB,cone,r,chord,twist,polars,
                        rho=rho,KinVisc=KinVisc,bTIDrag=False,bAIDrag=True,
                        a_init =a0, ap_init=ap0)
            # Export radial data to file
            filenameRadial = os.path.join(OutDir, '{}_BEM_ws{:02.0f}_x0={}_blockage_{}.csv'.format(run_name, V0, x0, blockage))
            BEM.WriteRadialFile(filenameRadial)
            print('>>>', filenameRadial)
            dfOut = BEM.StoreIntegratedValues(dfOut)
            # Export run data to excel file
            run_data = pd.DataFrame([run_name, u_turb_avg_percent, u_turb_min_percent, u_turb_max_percent, BEM.CP, BEM.CQ, BEM.CT, BEM.Edge, BEM.Flap])
            run_data = run_data.transpose()
            update_excel(run_data, os.path.join(OutDir, excel_name))
        # if type == 'rotor':
        #     # --- Export integrated values to file
        #     filenameOut = os.path.join(OutDir, 'Suite3_geo1_BEM_IntegratedValues_x0={}__blockage_{}_polar7.csv'.format(x0, blockage))
        #     dfOut.to_csv(filenameOut,sep='\t',index=False)
        #     print('>>>',filenameOut)
    return dfOut


if __name__=="__main__":
    OutDir                 = '_dataTmp'
    MainFASTFile           = '../data/BAR-WT/RotorSE_FAST_BAR_005a.fst'
    OperatingConditionFile = '../data/BAR_Rigid_Performances_FST.csv'

    #  --- Reading turbine operating conditions, Pitch, RPM, WS  (From FAST)
    df=weio.read(OperatingConditionFile).toDataFrame()
    Pitch = df['BldPitch_[deg]'].values
    Omega = df['RotSpeed_[rpm]'].values
    WS    = df['WS_[m/s]'].values

    dfOut = runParametricBEM_FAST(MainFASTFile, Pitch, Omega, WS)
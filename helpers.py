import pandas as pd
import numpy as np
from wake_model import get_u_turb
import scipy as sp
from scipy import optimize
from openpyxl import load_workbook

def read_excel(filepath, x0, nacelle_length, z0, Uinf):
    """ Read CFD excel file into a dataframe with relevant fields for plotting/comparison"""
    #TODO check directions in paraview file!!!

    CFD = pd.read_csv(filepath)  # sheet_name="nacelle_blockage_line_"+str(x0))
    CFD.replace(to_replace=' nan', value=0, inplace=True)
    CFD.replace(to_replace=' nan ', value=0, inplace=True)
    CFD['y'] = (CFD['y']-z0)/nacelle_length
    CFD['x'] = x0
    # CFD.rename(columns={'velocity_:0': 'U_local', 'velocity_:1': 'V_local'})
    CFD['WS'] = CFD['vmag'].astype(float)
    CFD['WS_nondim'] = CFD['WS'].divide(Uinf)
    CFD['u_turb'] = (CFD['vmagNormal'])
    return CFD

def diff(ab, nacelle_L, x_0, Uinf, Cd, wake, CFD_path, BEM, export):
    """find difference in wind speed between ellipsoid eng model and rectangular CFD model"""
    a = ab[0]
    b = ab[1]
    r = np.linspace(0, 105, 1000)
    avgDiff = np.zeros((3, 1))
    u_turb_EM = get_u_turb(x0=x_0, r=r, cone=0, Uinf=Uinf, Cd=0.2, a=a, b=b, wake=False, type=
    'rotor', plot=0, export=False)
    print(sum(u_turb_EM))
    CFD = pd.read_csv(CFD_path, skiprows=[0, 1, 2, 3, 5])
    # CFD = pd.read_excel("../data/y0/nacelle_blockage_LES.xls", sheet_name=str(x_0[i]) + "_comparison_data")
    r_CFD = CFD.loc[:, "Y [ m ]"].to_numpy() / nacelle_L
    u_CFD = CFD.loc[:, "Velocity u [ m s^-1 ]"].to_numpy()
    u_turb_CFD = (u_CFD - Uinf)/Uinf
    u_turb_CFD = np.insert(u_turb_CFD, 0, -1)
    r_CFD = np.insert(r_CFD, 0, 0)
    f = sp.interpolate.interp1d(r_CFD, u_turb_CFD)
    u_turb_CFD = f(r/nacelle_L)
    # r1 = 0.5*nacelle_L
    # r2 = 1.5*nacelle_L
    # diff = 0
    # for j in range(0, len(r)):
    #     if r[j] >= r1:
    #         startind = j
    #         break
    # for j in range(0, len(r)):
    #     if r[j] >= r2:
    #         endind = j
    #         break
    # diff = sum(abs(u_turb_EM[startind:endind]-u_turb_CFD[startind:endind]))
    #         # avgDiff[i] = diff/(endind-startind-1)/Uinf
    diff = sum(abs(u_turb_EM - u_turb_CFD))
    avgDiff = diff / len(u_turb_CFD)
    u_avg_CFD = np.mean(u_turb_CFD)
    u_avg_EM = np.mean(u_turb_EM)
    u_max_CFD = np.max(u_turb_CFD)
    u_max_EM = np.max(u_turb_EM)
    avgNormDiff = abs(avgDiff/u_avg_CFD)

    if BEM:
        BEM = MiniBEM(RPM, pitch, V0, xdot, u_turb,
                      nB, cone, r, chord, twist, polars,
                      rho=rho, KinVisc=KinVisc, bTIDrag=False, bAIDrag=True,
                      a_init=a0, ap_init=ap0)

    # if export:
    #     update_excel(data, path=)

    return avgDiff


def update_excel(data, path):
    """Outputs data from run to excel file"""
    writer = pd.ExcelWriter(path, engine='openpyxl')
    writer.book = load_workbook(path)
    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
    reader = pd.read_excel(path)
    a= len(reader)
    data.to_excel(writer, index=False, header=False, startrow=len(reader) + 1, startcol=0)
    writer.close()
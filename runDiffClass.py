# --- Common libraries
import os
import numpy as np
import pandas as pd
import scipy as sp
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
# --- Local libraries
import weio
from MiniBEM import MiniBEM, FASTFile2MiniBEM
from SourceEllipsoid import ser_u
import time




class runDiff():

    def __init__(self):
        """ setup BEM """
        self.BEM = True  # flag to run BEM
        self.OutDir = '_dataTmp'
        self.MainFASTFile = '../data/BAR-WT/RotorSE_FAST_BAR_005a.fst'
        self.OperatingConditionFile = '../data/BAR_Rigid_Performances_FST.csv'
        #  --- Reading turbine operating conditions, Pitch, RPM, WS  (From FAST)
        self.nB, self.cone, self.rBEM, self.chord, self.twist, self.polars, self.rho, self.KinVisc = FASTFile2MiniBEM(self.MainFASTFile)
        self.df = weio.read(self.OperatingConditionFile).toDataFrame()
        self.Pitch = self.df['BldPitch_[deg]'].values
        self.Omega = self.df['RotSpeed_[rpm]'].values
        self.WS = self.df['WS_[m/s]'].values
        # 5 corresponds to Uinf = 8 m/s 0-3, 5-8, 12-15, 22-25
        self.Uinf = self.WS[5]  # [m/s]
        self.RPM = self.Omega[5]  # [rpm]
        self.pitch = self.Pitch[5]  # [deg]
        self.u_turb = 0  # [m/s]
        self.xdot = 0 # [m/s]
        self.cone = 0  # cone angle, degrees
        self.R = 103  # m
        """ end setup BEM """

        """ eng model setup"""
        self.nacelle_L = 20  # m
        self.wake = False
        self.r = np.linspace(0, 104, 1040)
        self.x0 = [-0.5, 0, 0.5]  # up, down, mid
        self.run = "S2G2"
        self.a = 1/2
        self.b = 1/4
        self.Cd = 0.054  # U=8. Ellipsoids [a/b Cd] [4 0.052], [2 0.054], [1 0.124], [2/3 0.280] Bullet20x30: 0.042; Rectangle: 0.711; Pill: 0.144; Ellipsoid20x10 [U Cd] [3 0.050] [15 0.043] [25 0.063]

        """ end eng model setup"""

        """CFD input"""
        # self.CFD_folder = "/Users/banderso2/Documents/Office/BAR/2019-nacelle-blockage/data/Fluent/user_files/"
        self.CFD_folder = "/Users/banderso2/Documents/Office/BAR/2019-nacelle-blockage/data/Nalu/"

        """initialize u_turb arrays"""
        self.u_turb_EM_fine = np.zeros((len(self.x0), len(self.r)))
        self.u_turb_CFD_fine = np.zeros((len(self.x0), len(self.r)))

        """ plot setup """
        self.font_size = 24

        """outputs"""
        self.type = "grid"
        self.plot = 2  # 0=no plot, 1=show plot, 2=save plot
        self.export = False
        self.save_turb = False
        # self.plot = 1  # 0=no plot, 1=show plot, 2=save plot
        # self.export = False
        # self.save_turb = False



    def diff(self):
        """find difference in wind speed between ellipsoid eng model and rectangular CFD model"""
        for i in range(0, len(self.x0)):
            avgDiff = np.zeros((3, 1))
            self.u_turb_EM_fine[i, :] = self.get_u_turb(self.r, i)
            t = time.time()
            self.u_turb_EM_BEM = self.get_u_turb(self.rBEM, i)
            elapsed = time.time() - t
            print(elapsed)
            # self.u_turb_EM_fine = self.get_u_turb(x0=self.x0[i], r=self.r, cone=0, Uinf=self.Uinf, Cd=self.Cd, a=self.a, b=self.b, wake=self.wake, type=
            # 'rotor', plot=self.plot, export=self.export)
            # u_turb_EM_BEM = self.get_u_turb(x0=self.x0[i], r=self.rBEM, cone=0, Uinf=self.Uinf, Cd=self.Cd, a=self.a, b=self.b,
            #                             wake=self.wake, type=
            #                             'rotor', plot=self.plot, export=self.export)
            self.CFD_path = self.CFD_folder+self.run+"x"+str(self.x0[i])+".csv"
            # CFD = pd.read_csv(self.CFD_path, skiprows=[0, 1, 2, 3, 4], na_values=[' null', ' nan'])
            CFD = pd.read_csv(self.CFD_path, na_values=[' null', ' nan', ' nan '])
            CFD = CFD.fillna(0)
            # CFD = pd.read_excel("../data/y0/nacelle_blockage_LES.xls", sheet_name=str(x_0[i]) + "_comparison_data")
            # self.r_CFD = CFD.loc[:, " Y [ m ]"].to_numpy() / self.nacelle_L
            # u_CFD = CFD.loc[:, " Velocity u [ m s^-1 ]"].to_numpy()
            self.r_CFD = CFD.loc[:, "y"].to_numpy() / self.nacelle_L
            u_CFD = CFD.loc[:, "vX"].to_numpy()
            # u_CFD = np.where(u_CFD > 0, u_CFD, 0)  # zero out negative velocities
            u_turb_CFD = (u_CFD - self.Uinf) / self.Uinf
            u_turb_CFD = np.insert(u_turb_CFD, 0, -1)
            self.r_CFD = np.insert(self.r_CFD, 0, 0)
            f = sp.interpolate.interp1d(self.r_CFD, u_turb_CFD)
            self.u_turb_CFD_fine[i, :] = f(self.r / self.nacelle_L)
            self.u_turb_CFD_BEM = f(self.rBEM / self.nacelle_L)
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

            u_abs_avg_CFD = np.mean(abs(self.u_turb_CFD_fine[i, :]))
            u_abs_avg_EM = np.mean(abs(self.u_turb_EM_fine[i, :]))
            avg_diff = u_abs_avg_EM-u_abs_avg_CFD
            u_max_CFD = np.max(self.u_turb_CFD_fine[i, :])
            u_max_EM = np.max(self.u_turb_EM_fine[i, :])
            avg_norm_pDiff = avg_diff/u_abs_avg_CFD * 100

            if self.BEM:
                # self.u_turb_EM_BEM = 0
                EM_BEM = MiniBEM(self.RPM, self.pitch, self.Uinf, self.xdot, self.u_turb_EM_BEM,
                              self.nB, self.cone, self.rBEM, self.chord, self.twist, self.polars,
                              rho=self.rho, KinVisc=self.KinVisc, bTIDrag=False, bAIDrag=True,
                              a_init=False, ap_init=False)
                CFD_BEM = MiniBEM(self.RPM, self.pitch, self.Uinf, self.xdot, self.u_turb_CFD_BEM,
                              self.nB, self.cone, self.rBEM, self.chord, self.twist, self.polars,
                              rho=self.rho, KinVisc=self.KinVisc, bTIDrag=False, bAIDrag=True,
                              a_init=False, ap_init=False)



            if self.export:
                 run_data = pd.DataFrame([self.run, u_abs_avg_CFD, u_max_CFD, CFD_BEM.CP, CFD_BEM.CQ, CFD_BEM.CT,
                      CFD_BEM.Edge, CFD_BEM.Flap, 0, 0, 0, 0, 0, u_abs_avg_EM, u_max_EM, EM_BEM.CP, EM_BEM.CQ, EM_BEM.CT,
                      EM_BEM.Edge, EM_BEM.Flap, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                 self.run_data = run_data.transpose()
                 excel_name = "Nacelle_Blockage_comparison_x"+str(self.x0[i])+".xlsx"
                 self.excel_outpath = os.path.join(self.OutDir, excel_name)
                 self.update_excel()
        # TODO doing different line plotting...
        # if self.plot:
        #     self.plot_turb_lines(2)

        if self.save_turb:
            # self.save(self.u_turb_EM_fine, os.path.join(self.OutDir, 'u_turb_EM'+self.run+'.csv'))
            np.save(os.path.join(self.OutDir, 'turb/'+self.run+'u_turb_EM'), self.u_turb_EM_fine)
            np.save(os.path.join(self.OutDir, 'turb/'+self.run+'u_turb_CFD'), self.u_turb_CFD_fine)


        return avgDiff

    def update_excel(self):
        """Outputs data from run to excel file"""
        writer = pd.ExcelWriter(self.excel_outpath, engine='openpyxl')
        writer.book = load_workbook(self.excel_outpath)
        writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
        reader = pd.read_excel(self.excel_outpath)
        a = len(reader)
        self.run_data.to_excel(writer, index=False, header=False, startrow=len(reader) + 1, startcol=0)
        writer.close()



    def get_u_turb(self, r=0, i=0):
        """ returns the fractional turbulence, u_turb, along the specified rotor line or grid
        INPUTS:
        r: numpy vector representing radial locations considered
        i: index representing x0 value (which radial line considered)"""
        """INPUTS"""

        self.cone = self.cone * np.pi / 180

        # non-dimensionalize stuff
        r = r / self.nacelle_L

        if self.type == 'rotor':
            self.y = r * np.cos(self.cone)
            self.x = self.x0[i] + abs(r) * np.sin(self.cone)  # rotor segment downwind distance from nacelle front
            self.xlim = max(self.x)
            self.ylim = max(self.y)  # max(y)
        elif self.type == 'grid':
            nx = 1000
            ny = 500
            self.xlim = 1
            self.ylim = 1
            x1 = np.linspace(-self.xlim, self.xlim, nx)
            y1 = np.linspace(0, self.ylim, ny)
            self.dx = x1[1] - x1[0]
            self.dy = y1[1] - y1[0]
            self.x, self.y = np.meshgrid(x1, y1)

        # D method
        self.x_wake = np.linspace(-self.a, self.x[-1] + self.Uinf, 1000)
        # quadratic method

        self.midline = np.zeros(1000)
        nacelle_phi = np.linspace(0, np.pi, num=1000)
        self.nacelle_x = self.a * np.cos(nacelle_phi)
        self.nacelle_y = self.b * np.sin(nacelle_phi)
        # self.rect_x = np.array([-0.5, -0.5, 0.5, 0.5])
        # self.rect_y = np.array([0, 3.188 / nacelle_length, 3.188 / nacelle_length, 0])
        # no wake
        # u = 1 - ((x + 0.1)**2 - y**2)/((x + 0.1)**2 + y**2)**2 + Cd/(2*np.pi)*(x + 0.1)/((x + 0.1)**2 + y**2)
        # v = 2*(x + 0.1)*y/((x + 0.1)**2 + y**2)**2 + Cd/(2*np.pi)*y/((x + 0.1)**2 + y**2)
        # U_out_wake = u*Uinf
        # V_out_wake = v*Uinf
        U, V_out_wake = ser_u(self.x, self.y, self.Uinf, self.a, self.b)  # using SourceEllipsoid
        U_out_wake = U + self.Uinf

        # wake
        if self.wake:
            wake_scale = 1
            self.x_wake = np.linspace(-self.a, self.x[-1] + self.Uinf, 1000)

            # D method
            d = ((self.x + self.a) ** 2 + self.y ** 2) ** 0.5
            # y_wake = np.sqrt(1 / 2 * (b ** 2 + np.sqrt(4 * (x_wake + a) ** 2 + b ** 4))) - b
            # quadratic method
            self.y_wake = wake_scale * self.x_wake ** 2
            u_wake = self.Cd / d ** 0.5 * np.cos(np.pi / 2 * (self.y / d ** 0.5)) ** 2
            U_in_wake = U_out_wake - u_wake * self.Uinf  # waked velocity includes potential flow around nacelle times wake effects
            U_local = np.where((abs(self.y) < d ** 0.5) & (self.x > 0), U_in_wake, U_out_wake)
        else:
            U_local = U_out_wake

        U_local = np.where((abs(self.x) < max(self.nacelle_x)) & (self.y < (2 * self.b * abs((1 / 4 - self.x ** 2 / (2 * self.a) ** 2)) ** 0.5)), 0,
                           U_local)
        V_local = np.where((abs(self.x) < max(self.nacelle_x)) & (self.y < (2 * self.b * abs((1 / 4 - self.x ** 2 / (2 * self.a) ** 2)) ** 0.5)), 0,
                           V_out_wake)
        self.WS = np.sqrt(U_local ** 2 + V_local ** 2)
        self.WS_nondim = self.WS / self.Uinf
        u_turb = (U_local - self.Uinf) / self.Uinf  # turbulence, fractional
        # u_turb = u_turb*0

        return u_turb


    def plot_turb_lines(self, i):
        """plots turbulence of EM and CFD along rotor"""
        plt.close()
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['font.size'] = self.font_size
        plt.plot(self.r, self.u_turb_EM_fine[i, :]*100)
        plt.plot(self.r, self.u_turb_CFD_fine[i, :]*100)
        plt.xlabel('r [m]')
        plt.ylabel(r'$U_{turb} (\%)$')
        plt.legend(('EM', 'CFD'), loc='right')
        if self.plot == 1:
            plt.show()
            plt.close()
        elif self.plot == 2:
            plot_name = "figures/"+self.run+"/_x"+str(self.x0[i])+".pdf"
            plt.savefig(os.path.join(self.OutDir, plot_name), bbox_inches='tight')

    # def save(self, dat, filename):
    #     """pickles EM and CFD turbulence data for future plotting"""
    #     with open(filename, 'wb') as f:
    #         pickle.dump(dat, f)
    #
    # def load(self, filename):
    #     with open(filename, 'rb') as f:
    #         dat = pickle.load(f)
    #     return dat


    def plot_all_turb_lines(self, i):
        """plots turbulence of EM and CFD along rotor"""
        plt.close()
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['font.size'] = self.font_size
        plt.plot(self.r, self.u_turb_EM_fine[i, :] * 100)
        plt.plot(self.r, self.u_turb_CFD_fine[i, :] * 100)
        plt.xlabel('r [m]')
        plt.ylabel(r'$U_{turb} (\%)$')
        plt.legend(('EM', 'CFD'), loc='right')
        if self.plot == 1:
            plt.show()
            plt.close()
        elif self.plot == 2:
            plot_name = "figures/" + self.run + "/_x" + str(self.x0[i]) + ".pdf"
            plt.savefig(os.path.join(self.OutDir, plot_name), bbox_inches='tight')


    def plot_turb_contour(self):
        """plots turbulence contour plot of EM"""
        font_size = 16
        plt.rcParams['figure.figsize'] = 10, 5
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Tahoma']
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['font.size'] = font_size
        r = np.linspace(0, 104, 1040) / self.R * 100
        plt.gca().set_aspect('equal')
        fig = plt.figure(1, figsize=(10, 10))
        plt.axis('square')
        self.WS_nondim = self.WS_nondim[:-1, :-1]
        levels = MaxNLocator(nbins=100).tick_values(self.WS_nondim.min(), self.WS_nondim.max())
        levels = np.arange(-0.21, 1.22, 0.01)
        cmap = plt.get_cmap('jet')
        cf = plt.contourf(self.x[:-1, :-1] + self.dx / 2.,
                          self.y[:-1, :-1] + self.dy / 2., self.WS_nondim, levels=levels,
                          cmap=cmap)
        # cf = plt.contourf(x + dx / 2.,
        #                   y + dy / 2., WS, levels=levels,
        #                   cmap=cmap)

        plt.fill_between(self.nacelle_x, self.nacelle_y, where=self.nacelle_y >= self.midline, interpolate=True, color='grey')
        # plt.plot(self.rect_x, self.rect_y, color='k')
        if self.wake:
            plt.plot(self.x_wake, self.y_wake, color='b')
        plt.clim(-0.21, 1.21)
        plt.colorbar(cf, fraction=0.0245, pad=0.02, ticks=[-0.21, -0.13, -0.04, 0.04, 0.12, 0.21, 0.29, 0.38, 0.46, 0.55, 0.63, 0.72, 0.80, 0.89, 0.97, 1.06, 1.14, 1.21])
        plt.tick_params(which='both')
        plt.axis([-self.xlim, self.xlim, 0, self.ylim])
        plt.axis('off')
        # plt.xlabel(r'$x/L_{nacelle}$')
        plt.ylabel(r'$y/L_{nacelle}$')
        if self.plot == 1:
            plt.show()
            plt.close()
        elif self.plot == 2:
            plot_name = "figures/"+self.run + "/contour.pdf"
            plt.savefig(os.path.join(self.OutDir, plot_name), bbox_inches='tight')




if __name__=="__main__":
    myrun = runDiff()
    if myrun.type == "rotor":
        myrun.diff()
    elif myrun.type == "grid":
        myrun.get_u_turb()
        if myrun.plot:
            myrun.plot_turb_contour()
    elif myrun.type == "both":
        myrun.type = "rotor"
        myrun.diff()
        myrun.type = "grid"
        myrun.get_u_turb()
        if myrun.plot:
            myrun.plot_turb_contour()



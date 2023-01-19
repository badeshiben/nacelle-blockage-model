""" Replicate AeroDyn tower wake model"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from SourceEllipsoid import ser_u


class WakeModel():

    def __init__(self,r, x0, cone, Uinf, Cd, a, b, plot):
        self.r = r
        self.x0 = x0
        self.cone = cone
        self.Uinf = Uinf
        self.Cd = Cd
        self.a = a
        self.b = b
        self.plot = plot


    def get_u_turb(r, x0, cone, Uinf, Cd, a, b, plot):
        """ returns the fractional turbulence, u_turb, along the specified rotor
        INPUTS:
        r: points defining rotor radius [np.array]
        x0: downstream distance of rotor base from tower center [m]
        cone: blade cone angle, downstream [deg]
        Uinf: free stream velocity [m/s]
        Cd: nacelle drag coefficient
        plot: flag for plotting. 2=yes, save figs 1=yes show figs, 0=no"""

        type = 'grid'
        # y = np.concatenate((-(np.flip(r)), r))
        cone = cone*np.pi/180
        """ rotor"""
        y = r*np.cos(cone)
        x = x0 + abs(r)*np.sin(cone) # rotor segment downwind distance from tower center
        xlim = max(x)+Uinf
        ylim = 20      # max(y)
        x_wake = np.linspace(0, x[-1] + Uinf, 100)
        y_wake = np.sqrt(1 / 2 * (b**2 + np.sqrt(4 * x_wake** 2 + b**4)))
        midline = np.zeros(len(y_wake))
        nacelle_phi = np.linspace(0, np.pi / 2, num=len(x_wake))
        """ grid """
        if type == 'grid':
            nx = 1000
            ny = 500
            nacelle_phi = np.linspace(0, np.pi, num=len(x_wake))
            x1 = np.linspace(-xlim, xlim, nx)
            y1 = np.linspace(0, ylim, ny)
            dx = x1[1]-x1[0]
            dy = y1[1]-y1[0]
            x, y = np.meshgrid(x1, y1)

        nacelle_x = a * np.cos(nacelle_phi)
        nacelle_y = b * np.sin(nacelle_phi)
        # no wake
        # u = 1 - ((x + 0.1)**2 - y**2)/((x + 0.1)**2 + y**2)**2 + Cd/(2*np.pi)*(x + 0.1)/((x + 0.1)**2 + y**2)
        # v = 2*(x + 0.1)*y/((x + 0.1)**2 + y**2)**2 + Cd/(2*np.pi)*y/((x + 0.1)**2 + y**2)
        # U_out_wake = u*Uinf
        # V_out_wake = v*Uinf
        U, V_out_wake = ser_u(x, y, Uinf, a, b)  # using SourceEllipsoid
        # U_out_wake = np.ones(np.shape(U))*Uinf
        U_out_wake = U+Uinf
        # U_out_wake = U_out_wake*0

        # wake
        d = ((x+a)**2+y**2)**0.5
        u_wake = Cd/d**0.5*np.cos(np.pi/2*(y/d**0.5))**2
        U_in_wake = U_out_wake-u_wake*Uinf  # waked velocity includes potential flow around nacelle times wake effects
        # U_in_wake = U_out_wake

        U_local = np.where((abs(y) < d**0.5) & (x > 0), U_in_wake, U_out_wake)
        U_local = np.where((abs(x) < max(nacelle_x)) & (y < (2*b*abs((1/4 - x**2/(2*a)**2))**0.5)), 0, U_local)
        V_local = np.where((abs(x) < max(nacelle_x)) & (y < (2*b*abs((1/4 - x**2/(2*a)**2))**0.5)), 0, 0)
        WS = np.sqrt(U_local**2+V_local**2)
        print(WS)
        u_turb = (U_local-Uinf)/Uinf  # turbulence, fractional

        if plot:
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman']
            font = '24 '
            plt.rcParams.update({'font.size': font})
            plt.rcParams['font.size'] = font
            if type == 'rotor':
                plt.gca().set_aspect('equal', adjustable='box')
                plt.figure(1, figsize=(10, 10))

                plt.fill_between(x_wake, y_wake, where=y_wake >= midline, interpolate=True, color='blue')
                plt.fill_between(nacelle_x, nacelle_y, where=nacelle_y >= midline, interpolate=True, color='grey')
                # wake_bound = plt.plot(x_wake, y_wake, color='b')
                # wake_bound2 = plt.plot(x_wake, -y_wake, color='b', label='_nolegend_')
                plt.plot(x, y, color='k', linewidth=2)
                plt.quiver(x, y, U_local, V_local, angles='xy', scale_units='xy', scale=1,
                                        color='r', label='Data', linewidths=100)
                plt.xlim((0, max(x)+Uinf))
                plt.ylim((0, max(y)))
                # plt.legend(('rotor', 'wake', 'nacelle', 'wind velocity'), loc='right')
                plt.tick_params(which='both')
                plt.xlabel('x (m)')
                plt.ylabel('y (m)')
                # plt.title('Wind Velocity [m/s]')
                if plot == 2:
                    z=1
                    # plt.savefig('_dataTmp/figures/rotor_WS='+str(int(Uinf))+'.pdf')
                elif plot == 1:
                    plt.show()
                    plt.close()
            elif type == 'grid':
                # """ quiver plot """
                # plt.gca().set_aspect('equal', adjustable='box')
                # plt.figure(1, figsize=(10, 10))
                # plt.axis('square')
                # plt.frameon=False
                # midline = np.zeros(len(y_wake))
                # nacelle_phi = np.linspace(0, np.pi, num=len(x_wake))
                # nacelle_x = a*np.cos(nacelle_phi)
                # nacelle_y = b*np.sin(nacelle_phi)
                # plt.fill_between(x_wake, y_wake, where=y_wake >= midline, interpolate=True, color='blue')
                # plt.fill_between(nacelle_x, nacelle_y, where=nacelle_y >= midline, interpolate=True, color='grey')
                # # wake_bound = plt.plot(x_wake, y_wake, color='b')
                # # wake_bound2 = plt.plot(x_wake, -y_wake, color='b', label='_nolegend_')
                # #plt.plot(x, y, color='k', linewidth=2)
                # plt.quiver(x, y, U_local/Uinf, V_local/Uinf, angles='xy', scale_units='xy', scale=2,
                #                         color='r', label='Data', linewidths=100)
                # plt.xlim((-xlim, xlim))
                # plt.ylim((0, ylim))
                # # plt.legend(('rotor', 'wake', 'nacelle', 'wind velocity'), loc='right')
                # plt.tick_params(which='both')
                # plt.xlabel('x (m)')
                # plt.ylabel('y (m)')
                # # plt.title('Wind Velocity, Uinf = '+str(Uinf)+' [m/s]')
                # if plot == 2:
                #     zz=1
                #     plt.savefig('_dataTmp/figures/field_quiver_WS='+str(int(Uinf))+'.pdf', bbox_inches='tight')
                # elif plot == 1:
                #     plt.show()
                #     plt.close()
                """ contour plot"""
                plt.gca().set_aspect('equal')
                fig = plt.figure(2, figsize=(10, 10))
                plt.axis('square')
                WS = np.sqrt(np.square(U_local) + np.square(V_local))
                WS = WS/Uinf  # normalize for plot
                WS = WS[:-1, :-1]
                levels = MaxNLocator(nbins=100).tick_values(WS.min(), WS.max())
                cmap = plt.get_cmap('viridis')
                cf = plt.contourf(x[:-1, :-1] + dx / 2.,
                                  y[:-1, :-1] + dy / 2., WS, levels=levels,
                                  cmap=cmap)
                # cf = plt.contourf(x + dx / 2.,
                #                   y + dy / 2., WS, levels=levels,
                #                   cmap=cmap)

                plt.fill_between(nacelle_x, nacelle_y, where=nacelle_y >= midline, interpolate=True, color='grey')
                plt.colorbar(cf, fraction=0.0245, pad=0.02)
                plt.tick_params(which='both')
                plt.axis([-xlim, xlim, 0, ylim])
                plt.xlabel('x (m)')
                plt.ylabel('y (m)')
                #plt.title('Wind Speed, '+'$\mathregular{U_{inf}}$'+' [m/s]')
                if plot == 2:
                    zz=1
                    plt.savefig('_dataTmp/figures/contour_WS=' + str(int(Uinf)) + '.pdf', bbox_inches='tight')
                elif plot == 1:
                    plt.show()
                    plt.close()
        return u_turb
    """ only include with grid """
    # plot color contours


if __name__=="__main__":
    Cd = 1
    Uinf = 10;

    """if create whole grid"""
    # dx = 0.2
    # dy = 0.2
    # x0 = 0
    # xend = 6
    # y0 = -3
    # yend = 3
    # nx = int((xend - x0)/dx + 1)
    # ny = int((yend - x0)/dy + 1)
    # #meshgrid
    # xl = np.linspace(x0, xend, nx)
    # yl = np.linspace(y0, yend, ny)
    # x, y = np.meshgrid(xl, yl)

    """if just interested in rotor line"""
    y = np.linspace(0, 20, num=21)
    cone = np.pi / 18  # cone angle in radians
    x0 = 10

    get_u_turb(y, x0, cone, Uinf, Cd)
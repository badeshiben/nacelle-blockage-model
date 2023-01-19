
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator

""" linestyles """
linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

def compare_TI():
    """ Plot turbulence data from multiple locations on one figure """
    plot_path = '_dataTmp/figures/'
    plot_name = 'diff_TI.pdf'
    nlocs = 3
    plot = 1  # flag 1=show, 2=save
    plt.close()
    font_size = 12
    plt.rcParams['figure.figsize'] = 8, 5
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.size'] = font_size
    R = 103
    r = np.linspace(0, 104, 1040) / R * 100
    data1 = np.random.rand(nlocs, len(r))
    data2 = np.random.rand(nlocs, len(r))
    # data1 = np.load('_dataTmp/turb/'+run+'u_turb_EM.npy')
    # data2 = np.load('_dataTmp/turb/'+run+'u_turb_CFD.npy')
    data1 = np.load('_dataTmp/turb/S1T1Bu_turb_CFD.npy')
    data2 = np.load('_dataTmp/turb/S1T2Bu_turb_CFD.npy')
    data3 = np.load('_dataTmp/turb/S1T3Bu_turb_CFD.npy')
    data4 = np.load('_dataTmp/turb/S1T1Fu_turb_CFD.npy')
    data5 = np.load('_dataTmp/turb/S1T2Fu_turb_CFD.npy')
    data6 = np.load('_dataTmp/turb/S1T3Fu_turb_CFD.npy')
    yline = [0, 105]
    spacer = 60
    ticklist = []
    scaler = 35

    for i in range(0, nlocs):
        plt.plot(data1[i, :] * scaler + spacer * i, r, color='red')
        plt.plot(data2[i, :] * scaler + spacer * i, r, color='blue')
        plt.plot(data3[i, :] * scaler + spacer * i, r, color='green')
        plt.plot(data4[i, :] * scaler + spacer * i, r, color='xkcd:light pink', linestyle=(0, (5, 5)))
        plt.plot(data5[i, :] * scaler + spacer * i, r, color='xkcd:sky blue', linestyle=(0, (5, 5)))
        plt.plot(data6[i, :] * scaler + spacer * i, r, color='xkcd:light green', linestyle=(0, (5, 5)))
        plt.plot([spacer * i, spacer * i], yline, color='black', linestyle=(0, (1, 5)))
        plt.plot([spacer * i, spacer * i], yline, color='black', linestyle=(0, (1, 5)))
        plt.plot([spacer * i + 0.2 * scaler, spacer * i + 0.2 * scaler], yline, color='black', linestyle=(0, (1, 5)))
        ticklist.append(spacer * i - scaler)
        ticklist.append(spacer * i)
        ticklist.append(spacer * i + 0.2 * scaler)
    plt.ylim(0, 15)
    plt.ylabel('r/R %')
    plt.xlabel(r'$U/U_{0}$')
    plt.xticks(ticklist, ['0', '1', '1.2', '0', '1', '1.2', '0', '1', '1.2'])
    # plt.legend(('EM', 'CFD'), loc='upper right')
    plt.legend(('TI=1%', 'TI=5%', 'TI=10%'), fontsize=10, bbox_to_anchor=(0.75, 0.7))

    if plot == 1:
        plt.show()
        plt.close()
    elif plot == 2:
        plt.savefig(plot_path + plot_name, bbox_inches='tight')

def nacelles():
    plot_path = '../../2019-nacelle-blockage-12-17/article/figs/'
    plot_name = 'geometries.pdf'
    nlocs = 3
    plot = 1  # flag 1=show, 2=save
    plt.close()
    font_size = 10
    plt.rcParams['figure.figsize'] = 8, 5
    # plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Tahoma']
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.size'] = font_size
    nx = 1000
    a = [1/2, 1/2, 1/2, 1/2]#, 3/4, 1, 1]
    b = [1/8, 1/4, 1/2, 3/4]#, 1.34/4, 3/4, 1/2]
    pill_th = np.linspace(0, np.pi/2, nx/2)
    pill_y1 = np.sin(pill_th)/4
    pill_x1 = -np.cos(pill_th)/4 + 1/4
    pill_y2 = np.cos(pill_th)/4
    pill_x2 = np.sin(pill_th)/4 + 3/4
    pill_x = np.concatenate((pill_x1, pill_x2))
    pill_y = np.concatenate((pill_y1, pill_y2))
    bullet_x = -np.cos(pill_th)*1/2 + 1/2
    bullet_y = np.sin(pill_th)*3/4
    bullet_x = np.append(bullet_x, [1, 1])
    bullet_y = np.append(bullet_y, [3/4, 0])
    rect_x = np.linspace(0, 1, nx-2)
    rect_x = np.insert(rect_x, 0, 0)
    rect_x = np.insert(rect_x, -1, 1)
    rect_y = np.ones(nx)/4
    rect_y = np.insert(rect_y, 0, 0)
    rect_y = np.insert(rect_y, -1, 0)
    nacelle_phi = np.linspace(0, np.pi, nx)
    nacelle_x = np.outer(np.transpose(np.cos(nacelle_phi)), a) + 1/2
    nacelle_y = np.outer(np.transpose(np.sin(nacelle_phi)), b)
    rect_EM_x = np.cos(nacelle_phi)*3/4 + 1/2
    rect_EM_y = np.sin(nacelle_phi)*1.34/4
    pill_EM_x = nacelle_x[:, 1]
    pill_EM_y = nacelle_y[:, 1]
    bullet_EM_x = np.cos(nacelle_phi) + 1
    bullet_EM_y = np.sin(nacelle_phi)*3/4
    yline = [0, 105]
    spacer = 10
    ticklist = []
    scaler = 5

    plt.plot(bullet_x, bullet_y, color='green')
    plt.plot(bullet_EM_x, bullet_EM_y, color='green', linestyle=(0, (5, 5)))
    plt.plot(rect_x, rect_y, color='red')
    plt.plot(rect_EM_x, rect_EM_y, color='red', linestyle=(0, (5, 5)))
    plt.plot(pill_x, pill_y, color='blue')
    plt.plot(pill_EM_x, pill_EM_y, color='orange')

    for i in range(0, len(a)):
        plt.plot(nacelle_x[:, i], nacelle_y[:, i], color='black', linestyle=(0, (5, 5)))

    plt.ylabel(r'$y/L_{nacelle}$')
    plt.xlabel(r'$x/L_{nacelle}$')
    plt.ylim(0, 0.8)
    plt.xlim(-0.25, 2.0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(('bullet', 'bullet & 20x30m ellipsoid EM', 'rectangle', 'rectangle EM', 'pill', 'pill EM', 'ellipsoids'), fontsize=10,
               loc='upper right')

    if plot == 1:
        plt.show()
        plt.close()
    elif plot == 2:
        plt.savefig(plot_path + plot_name, bbox_inches='tight')

def plot_Re_stuff():
    data = pd.read_excel('_dataTmp/Nacelle_blockage_comparison_x0.5.xlsx', sheet_name='Re graphs', usecols="A:J")
    plot_path = '../../../2019-nacelle-blockage-12-17/article/figs/'
    nacelle_L=20
    R = 103
    u_turb = data['u_turb'] + 1
    BLH = data['BLH']/nacelle_L
    ReL = data['Re_L']
    ReH = data['Re_H']
    AR = ReL/ReH  # aspect ratio l/h
    dCP = data['%dCP']*100
    plot = 2  # flag 1=show, 2=save
    plt.close()
    font_size = 10
    plt.rcParams['figure.figsize'] = 8, 2.5
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Tahoma']
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.size'] = font_size

    f1, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    plot_name = 'BLH_vs_Re.pdf'
    # ax1.plot(BLH[0:6], ReL[0:6], color='red')
    # ax2.plot(BLH[0:6], ReH[0:6], color='red')
    ax1.scatter(ReL[0:7], BLH[0:7], color='red')
    ax1.scatter(ReL[7], BLH[7], color='blue', marker='s')
    ax1.scatter(ReL[8], BLH[8], color='green', marker='D')
    ax1.scatter(ReL[9], BLH[9], color='xkcd:orange', marker='<')

    ax2.scatter(ReH[0:7], BLH[0:7], color='red')
    ax2.scatter(ReH[7], BLH[7], color='blue', marker='s')
    ax2.scatter(ReH[8], BLH[8], color='green', marker='D')
    ax2.scatter(ReH[9], BLH[9], color='xkcd:orange', marker='<')

    ax3.scatter(AR[0:7], BLH[0:7], color='red')
    ax3.scatter(AR[7], BLH[7], color='blue', marker='s')
    ax3.scatter(AR[8], BLH[8], color='green', marker='D')
    ax3.scatter(AR[9], BLH[9], color='xkcd:orange', marker='<')

    ax1.set_ylabel(r'$BLH/L_{nacelle}$')
    ax1.set_xlabel(r'$Re_{L}$')
    ax2.set_xlabel(r'$Re_{H}$')
    ax3.set_xlabel(r'$Aspect Ratio$')
    plt.legend(('ellipsoid', 'rectangle', 'pill', 'bullet'),  fontsize=10, bbox_to_anchor=(0.43, 0.5))
    if plot == 2:
        plt.savefig(plot_path + plot_name, bbox_inches='tight')
    if plot == 1:
        plt.show()

    f2, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    plot_name = 'max_speedup_vs_Re.pdf'
    # ax1.plot(BLH[0:6], ReL[0:6], color='red')
    # ax2.plot(BLH[0:6], ReH[0:6], color='red')
    ax1.scatter(ReL[0:7], u_turb[0:7], color='red')
    ax1.scatter(ReL[7], u_turb[7], color='blue', marker='s')
    ax1.scatter(ReL[8], u_turb[8], color='green', marker='D')
    ax1.scatter(ReL[9], u_turb[9], color='xkcd:orange', marker='<')

    ax2.scatter(ReH[0:7], u_turb[0:7], color='red')
    ax2.scatter(ReH[7], u_turb[7], color='blue', marker='s')
    ax2.scatter(ReH[8], u_turb[8], color='green', marker='D')
    ax2.scatter(ReH[9], u_turb[9], color='xkcd:orange', marker='<')

    ax3.scatter(AR[0:7], u_turb[0:7], color='red')
    ax3.scatter(AR[7], u_turb[7], color='blue', marker='s')
    ax3.scatter(AR[8], u_turb[8], color='green', marker='D')
    ax3.scatter(AR[9], u_turb[9], color='xkcd:orange', marker='<')
    ax1.set_ylabel((r'$max V/V_{0}$'))
    ax1.set_xlabel(r'$Re_{L}$')
    ax2.set_xlabel(r'$Re_{H}$')
    ax3.set_xlabel(r'$Aspect Ratio$')
    plt.legend(('ellipsoid', 'rectangle', 'pill', 'bullet'),  fontsize=10, bbox_to_anchor=(0.36, 1.02))

    if plot == 2:
        plt.savefig(plot_path + plot_name, bbox_inches='tight')
    if plot == 1:
        plt.show()

    f3, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    plot_name = 'dCp_vs_Re.pdf'
    # ax1.plot(BLH[0:6], ReL[0:6], color='red')
    # ax2.plot(BLH[0:6], ReH[0:6], color='red')
    ax1.scatter(ReL[0:7], dCP[0:7], color='red')
    ax1.scatter(ReL[7], dCP[7], color='blue', marker='s')
    ax1.scatter(ReL[8], dCP[8], color='green', marker='D')
    ax1.scatter(ReL[9], dCP[9], color='xkcd:orange', marker='<')

    ax2.scatter(ReH[0:7], dCP[0:7], color='red')
    ax2.scatter(ReH[7], dCP[7], color='blue', marker='s')
    ax2.scatter(ReH[8], dCP[8], color='green', marker='D')
    ax2.scatter(ReH[9], dCP[9], color='xkcd:orange', marker='<')

    ax3.scatter(AR[0:7], dCP[0:7], color='red')
    ax3.scatter(AR[7], dCP[7], color='blue', marker='s')
    ax3.scatter(AR[8], dCP[8], color='green', marker='D')
    ax3.scatter(AR[9], dCP[9], color='xkcd:orange', marker='<')
    ax1.set_ylabel((r'$\%\Delta C_{p}$'))
    ax1.set_xlabel(r'$Re_{L}$')
    ax2.set_xlabel(r'$Re_{H}$')
    ax3.set_xlabel(r'$Aspect Ratio$')
    plt.legend(('ellipsoid', 'rectangle', 'pill', 'bullet'),  fontsize=10, bbox_to_anchor=(0.35, 1.02))
    plt.ylim(0, 0.65)
    if plot == 2:
        plt.savefig(plot_path + plot_name, bbox_inches='tight')
    if plot == 1:
        plt.show()
        plt.close()

def get_BLH(case):
    u_turb = np.load(case)
    u_turb = u_turb[2, :]
    max_i = np.argmax(u_turb)
    u_t_max = u_turb[max_i]
    u_nondim = u_turb + 1
    BLu = (1+u_t_max)*0.99
    for i in range(0, len(u_nondim)):
        if u_nondim[i] >= BLu:
            BLH = i/10
            break
    print('BLH from midline for '+case+'= '+str(BLH))
    print('u_turb_max for       '+case+'= '+str(u_t_max))

def compare_U():
    """ Plot turbulence data from multiple locations on one figure """
    plot_path = '_dataTmp/figures/'
    plot_name = 'diff_U_EM.pdf'
    nlocs = 3
    plot = 2  # flag 1=show, 2=save
    plt.close()
    font_size = 12
    plt.rcParams['figure.figsize'] = 8, 5
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.size'] = font_size
    R = 103
    r = np.linspace(0, 104, 1040)/R*100
    data1 = np.random.rand(nlocs, len(r))
    data2 = np.random.rand(nlocs, len(r))
    # data1 = np.load('_dataTmp/turb/'+run+'u_turb_EM.npy')
    # data2 = np.load('_dataTmp/turb/'+run+'u_turb_CFD.npy')
    data1 = np.load('_dataTmp/turb/S4U1u_turb_CFD.npy')
    data2 = np.load('_dataTmp/turb/S4U2u_turb_CFD.npy')
    data3 = np.load('_dataTmp/turb/S4U3u_turb_CFD.npy')
    data4 = np.load('_dataTmp/turb/S4U1u_turb_EM.npy')
    data5 = np.load('_dataTmp/turb/S4U2u_turb_EM.npy')
    data6 = np.load('_dataTmp/turb/S4U3u_turb_EM.npy')
    yline = [0, 105]
    spacer = 60
    ticklist = []
    scaler = 35

    for i in range(0, nlocs):
        plt.plot(data1[i, :] * scaler + spacer * i, r, color='red')
        plt.plot(data2[i, :] * scaler + spacer * i, r, color='blue')
        plt.plot(data3[i, :] * scaler + spacer * i, r, color='green')
        plt.plot(data4[i, :] * scaler + spacer * i, r, color='xkcd:orange', linestyle=(0, (5, 5)))
        # plt.plot(data5[i, :] * scaler + spacer * i, r, color='xkcd:sky blue', linestyle=(0, (1, 1)))
        # plt.plot(data6[i, :] * scaler + spacer * i, r, color='xkcd:light green', linestyle=(0, (5, 5)))
        plt.plot([spacer * i, spacer * i], yline, color='black', linestyle=(0, (1, 5)))
        plt.plot([spacer * i, spacer * i], yline, color='black', linestyle=(0, (1, 5)))
        plt.plot([spacer * i + 0.2 * scaler, spacer * i + 0.2 * scaler], yline, color='black', linestyle=(0, (1, 5)))
        ticklist.append(spacer * i - scaler)
        ticklist.append(spacer * i)
        ticklist.append(spacer * i + 0.2 * scaler)
    plt.ylim(0, 15)
    plt.ylabel('r/R %')
    plt.xlabel(r'$U/U_{0}$')
    plt.xticks(ticklist, ['0', '1', '1.2', '0', '1', '1.2', '0', '1', '1.2'])
    # plt.legend(('EM', 'CFD'), loc='upper right')
    plt.legend(('U=3 m/s', 'U=15 m/s', 'U=30 m/s', 'Eng Model'),  fontsize=10, bbox_to_anchor=(0.75, 0.7))

    if plot == 1:
        plt.show()
        plt.close()
    elif plot == 2:
        plt.savefig(plot_path + plot_name, bbox_inches='tight')

def compare_geos():
    """ Plot turbulence data from multiple locations on one figure """
    plot_path = '../../../2019-nacelle-blockage-12-17/article/figs/'
    plot_name = 'diff_geo_EM.pdf'
    nlocs = 3
    plot = 2  # flag 1=show, 2=save
    plt.close()
    font_size = 10
    plt.rcParams['figure.figsize'] = 8, 4
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Tahoma']
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.size'] = font_size

    yline = [0, 105]
    spacer = 10
    ticklist = []
    scaler = 5

    R = 103
    r = np.linspace(0, 104, 1040) / R*100
    nx = 1000
    midline = np.zeros(nx)
    nacelle_phi = np.linspace(0, np.pi, num=nx)

    rect_x = np.linspace(-10, 10, nx) + 5
    rect_y = np.ones(nx-2) * 5
    rect_y = np.insert(rect_y, 0, 0)
    rect_y = np.insert(rect_y, -1, 0) * 100/R

    pill_th = np.linspace(0, np.pi / 2, nx / 2)
    pill_y1 = np.sin(pill_th) * 5
    pill_x1 = -np.cos(pill_th) * 5
    pill_y2 = np.cos(pill_th) * 5
    pill_x2 = np.sin(pill_th) * 5 + 10
    pill_x = np.concatenate((pill_x1, pill_x2))
    pill_y = np.concatenate((pill_y1, pill_y2)) * 100/R
    a = 10
    b = 5
    ellipsoid_x = a * np.cos(nacelle_phi) + 5
    ellipsoid_y = b * np.sin(nacelle_phi) * 100/R

    plt.fill_between(rect_x, rect_y, where=rect_y >= midline, interpolate=True, color='xkcd:silver')
    plt.fill_between(pill_x, pill_y, where=pill_y >= midline, interpolate=True, color='grey')
    plt.fill_between(ellipsoid_x, ellipsoid_y, where=ellipsoid_y >= midline, interpolate=True, color='xkcd:slate grey')

    data1 = np.load('_dataTmp/turb/S3G2u_turb_CFD.npy')
    data2 = np.load('_dataTmp/turb/S2G1u_turb_CFD.npy')
    data3 = np.load('_dataTmp/turb/S2G2u_turb_CFD.npy')
    data4 = np.load('_dataTmp/turb/S3G2u_turb_EM.npy')
    data5 = np.load('_dataTmp/turb/S2G2u_turb_EM.npy')
    data6 = np.load('_dataTmp/turb/S2G1u_turb_EM.npy')


    for i in range(0, nlocs):
        plt.plot(data1[i, :] * scaler + spacer * i, r, color='red')
        plt.plot(data2[i, :] * scaler + spacer * i, r, color='blue')
        plt.plot(data3[i, :] * scaler + spacer * i, r, color='green')
        plt.plot(data4[i, :] * scaler + spacer * i, r, color='xkcd:pink')
        plt.plot(data5[i, :] * scaler + spacer * i, r, color='xkcd:sky blue', linestyle=(0, (5, 5)))
        plt.plot(data6[i, :] * scaler + spacer * i, r, color='xkcd:light green', linestyle=(0, (5, 5)))
        plt.plot([spacer * i, spacer * i], yline, color='black', linestyle=(0, (1, 5)))
        plt.plot([spacer * i, spacer * i], yline, color='black', linestyle=(0, (1, 5)))
        plt.plot([spacer * i + 0.2 * scaler, spacer * i + 0.2 * scaler], yline, color='black', linestyle=(0, (1, 5)))
        ticklist.append(spacer * i - scaler)
        ticklist.append(spacer * i)
        ticklist.append(spacer * i + 0.2 * scaler)
    plt.ylim(0, 12)
    plt.ylabel('r/R [%]')
    plt.xlabel(r'$V/V_{0}$')
    plt.xticks(ticklist, ['0', '1', '1.2', '0', '1', '1.2', '0', '1', '1.2'])
    # plt.legend(('EM', 'CFD'), loc='upper right')
    # plt.legend(('ellipsoid CFD', 'rectangle CFD', 'pill CFD', 'ellipsoid EM', 'rectangle EM', 'pill EM'), fontsize=10, bbox_to_anchor=(0.77, 0.65))
    plt.legend(('ellipsoid', 'rectangle', 'pill'), fontsize=10,
               bbox_to_anchor=(0.9, 0.78))
    plt.tight_layout(pad=0)

    if plot == 1:
        plt.show()
        plt.close()
    elif plot == 2:
        plt.savefig(plot_path + plot_name, bbox_inches='tight')

def compare_heights():
    """ Plot turbulence data from multiple locations on one figure """
    plot_path = '_dataTmp/figures/'
    plot_name ='diff_height.pdf'
    nlocs = 3
    plot = 1  # flag 1=show, 2=save
    plt.close()
    font_size = 12
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.size'] = font_size
    r = np.linspace(0, 104, 1040)
    data1 = np.random.rand(nlocs, len(r))
    data2 = np.random.rand(nlocs, len(r))
    # data1 = np.load('_dataTmp/turb/'+run+'u_turb_EM.npy')
    # data2 = np.load('_dataTmp/turb/'+run+'u_turb_CFD.npy')
    data1 = np.load('_dataTmp/turb/S3G1u_turb_CFD.npy')
    data2 = np.load('_dataTmp/turb/S3G2u_turb_CFD.npy')
    data3 = np.load('_dataTmp/turb/S3G3u_turb_CFD.npy')
    data4 = np.load('_dataTmp/turb/S3G4u_turb_CFD.npy')
    data5 = np.load('_dataTmp/turb/S3G5u_turb_CFD.npy')
    data5 = np.where(data5 < -1, -1, data5)
    data4 = np.where(data4 < -1, -1, data4)

    yline = [0, 105]
    spacer = 40
    ticklist = []
    scaler = 20

    for i in range(0, nlocs):
        plt.plot(data1[i, :] * scaler + spacer * i, r, color='red')
        plt.plot(data2[i, :] * scaler + spacer * i, r, color='blue')
        plt.plot(data3[i, :] * scaler + spacer * i, r, color='green')
        plt.plot(data4[i, :] * scaler + spacer * i, r, color='xkcd:orange')
        plt.plot(data5[i, :] * scaler + spacer * i, r, color='xkcd:purple')
        plt.plot([spacer * i, spacer * i], yline, color='black', linestyle=(0, (1, 5)))
        plt.plot([spacer * i, spacer * i], yline, color='black', linestyle=(0, (1, 5)))
        plt.plot([spacer * i + 0.2 * scaler, spacer * i + 0.2 * scaler], yline, color='black', linestyle=(0, (1, 5)))
        plt.plot([spacer * i + 0.4 * scaler, spacer * i + 0.4 * scaler], yline, color='black', linestyle=(0, (1, 5)))
        ticklist.append(spacer * i - scaler)
        ticklist.append(spacer * i)
        ticklist.append(spacer * i + 0.2 * scaler)
        ticklist.append(spacer * i + 0.4 * scaler)
    plt.ylim(0, 30)
    plt.ylabel('r [m]')
    plt.xlabel(r'$V/V_{0}$')
    plt.xticks(ticklist, ['0', '1', '1.2', '1.4', '0', '1', '1.2', '1.4', '0', '1', '1.2', '1.4'])
    # plt.legend(('EM', 'CFD'), loc='upper right')
    plt.legend(('e1', 'e2', 'e3', 'e4', 'b'), bbox_to_anchor=(0.7, 0.5))
    plt.tight_layout(pad=0)

def compare_heights_EM():
    """ Plot turbulence data from multiple locations on one figure """
    plot_path = '../../../2019-nacelle-blockage-12-17/article/figs/'
    plot_name = 'diff_height_EM.pdf'
    R = 103
    nlocs = 3
    plot = 2  # flag 1=show, 2=save
    plt.close()
    font_size = 12
    plt.rcParams['figure.figsize'] = 10, 5
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Tahoma']
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.size'] = font_size

    a = np.array([1/2, 1/2, 1/2, 1/2]) * 20
    b = np.array([3/4, 1/2, 1/4, 1/8]) * 20 *100/R
    nx = 1000
    midline = np.zeros(nx)
    nacelle_phi = np.linspace(0, np.pi, nx)
    nacelle_x = np.outer(np.transpose(np.cos(nacelle_phi)), a) + 5
    nacelle_y = np.outer(np.transpose(np.sin(nacelle_phi)), b)
    color = ['xkcd:greyish', 'grey', 'xkcd:slate grey', 'black']

    bullet_th = np.linspace(0, np.pi / 2, nx-2)
    bullet_x = -np.cos(bullet_th)*1/2 + 1/2
    bullet_y = np.sin(bullet_th)*3/4
    bullet_x = np.append(bullet_x, [1, 1]) * 20 - 5
    bullet_y = np.append(bullet_y, [3/4, 0]) * 20*100/R

    r = np.linspace(0, 104, 1040)/R * 100
    data1 = np.load('_dataTmp/turb/S3G1u_turb_CFD.npy')
    data2 = np.load('_dataTmp/turb/S3G2u_turb_CFD.npy')
    data3 = np.load('_dataTmp/turb/S3G3u_turb_CFD.npy')
    data4 = np.load('_dataTmp/turb/S3G4u_turb_CFD.npy')
    data5 = np.load('_dataTmp/turb/S3G5u_turb_CFD.npy')
    # data5 = np.where(data5 < -1, -1, data5)
    # data4 = np.where(data4 < -1, -1, data4)
    data6 = np.load('_dataTmp/turb/S3G1u_turb_EM.npy')
    data7 = np.load('_dataTmp/turb/S3G2u_turb_EM.npy')
    data8 = np.load('_dataTmp/turb/S3G3u_turb_EM.npy')
    data9 = np.load('_dataTmp/turb/S3G4u_turb_EM.npy')
    data10 = np.load('_dataTmp/turb/S3G5u_turb_EM.npy')
    yline = [0, 105]
    spacer = 10
    ticklist = []
    scaler = 5
    handlelist=[]

    plt.fill_between(bullet_x, bullet_y, where=bullet_y >= midline, interpolate=True,
                     color='xkcd:silver')
    for j in range(0, len(a)):
        plt.fill_between(nacelle_x[:, j], nacelle_y[:, j], where=nacelle_y[:, j] >= midline, interpolate=True, color=color[j])
    for i in range(0, nlocs):
        c1, = plt.plot(data1[i, :] * scaler + spacer * i, r, color='red')
        c2, = plt.plot(data2[i, :] * scaler + spacer * i, r, color='blue')
        c3, = plt.plot(data3[i, :] * scaler + spacer * i, r, color='green')
        c4, = plt.plot(data4[i, :] * scaler + spacer * i, r, color='xkcd:orange')
        c5, = plt.plot(data5[i, :] * scaler + spacer * i, r, color='xkcd:purple')
        e1, = plt.plot(data6[i, :] * scaler + spacer * i, r, color='xkcd:pink', linestyle=(0, (5, 5)))
        e2, = plt.plot(data7[i, :] * scaler + spacer * i, r, color='xkcd:sky blue', linestyle=(0, (5, 5)))
        e3, = plt.plot(data8[i, :] * scaler + spacer * i, r, color='xkcd:light green', linestyle=(0, (5, 5)))
        e4 = plt.plot(data9[i, :] * scaler + spacer * i, r, color='xkcd:light peach')
        e5 = plt.plot(data10[i, :] * scaler + spacer * i, r, color='xkcd:lavender', linestyle=(0, (5, 5)))
        plt.plot([spacer * i, spacer * i], yline, color='black', linestyle=(0, (1, 5)))
        plt.plot([spacer * i, spacer * i], yline, color='black', linestyle=(0, (1, 5)))
        plt.plot([spacer * i + 0.2 * scaler, spacer * i + 0.2 * scaler], yline, color='black', linestyle=(0, (1, 5)))
        plt.plot([spacer * i + 0.4 * scaler, spacer * i + 0.4 * scaler], yline, color='black', linestyle=(0, (1, 5)))
        ticklist.append(spacer * i - scaler)
        ticklist.append(spacer * i)
        ticklist.append(spacer * i + 0.2 * scaler)
        ticklist.append(spacer * i + 0.4 * scaler)
        # handlelist = handlelist +[c1, c2, c3, c4, c5, e1, e2, e3]
    plt.ylim(0, 30)
    plt.ylabel('r/R [%]')
    plt.xlabel(r'$V/V_{0}$')
    plt.xticks(ticklist, ['0', '1', '1.2', '1.4', '0', '1', '1.2', '1.4', '0', '1', '1.2', '1.4'])
    # plt.legend(('EM', 'CFD'), loc='upper right')
    plt.legend(('Ellipsoid 20x5', 'Ellipsoid 20x10', 'Ellipsoid 20x20', 'Ellipsoid 20x30', 'Bullet 20x30'), fontsize=10, bbox_to_anchor=(.87, .95))
    # plt.legend(('E20x5', 'E20x10', 'E20x20', 'E20x30', 'B20x30'), handles=[handlelist], fontsize=10, bbox_to_anchor=(0.2, 0.65))
    plt.tight_layout(pad=0)

    if plot == 1:
        plt.show()
        plt.close()
    elif plot == 2:
        plt.savefig(plot_path + plot_name, bbox_inches='tight')

def heights_CFDvsEM():
    """ Plot turbulence data from multiple locations on one figure """
    plot_path = '_dataTmp/figures/'
    run = 'S2G1'
    plot_name = run+'line.pdf'
    plot_name = run+'ellipsoid_CFD_VS_EM.pdf'
    nlocs = 3
    plot = 2  # flag 1=show, 2=save
    plt.close()
    font_size = 12
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.size'] = font_size
    r = np.linspace(0, 104, 1040)
    data1 = np.random.rand(nlocs, len(r))
    data2 = np.random.rand(nlocs, len(r))
    # data1 = np.load('_dataTmp/turb/'+run+'u_turb_EM.npy')
    # data2 = np.load('_dataTmp/turb/'+run+'u_turb_CFD.npy')
    data1 = np.load('_dataTmp/turb/S3G1u_turb_CFD.npy')
    data3 = np.load('_dataTmp/turb/S3G3u_turb_CFD.npy')
    data2 = np.load('_dataTmp/turb/S3G1u_turb_EM.npy')
    data4 = np.load('_dataTmp/turb/S3G3u_turb_EM.npy')
    # data4 = np.where(data4 < -1, -1, data4)


    yline = [0, 105]
    spacer = 40
    ticklist = []
    scaler = 20

    for i in range(0, nlocs):
        plt.plot(data1[i, :]*scaler+spacer*i, r, color='red')
        plt.plot(data2[i, :]*scaler+spacer*i, r, color='xkcd:pink')
        plt.plot(data3[i, :] * scaler + spacer * i, r, color='blue')
        plt.plot(data4[i, :] * scaler + spacer * i, r, color='xkcd:sky blue')
        plt.plot([spacer*i, spacer*i], yline, color='black', linestyle=(0, (1, 5)))
        plt.plot([spacer * i, spacer * i], yline, color='black', linestyle=(0, (1, 5)))
        plt.plot([spacer*i+0.2*scaler, spacer*i+0.2*scaler], yline, color='black', linestyle=(0, (1, 5)))
        plt.plot([spacer*i+0.4*scaler, spacer*i+0.4*scaler], yline, color='black', linestyle=(0, (1, 5)))
        ticklist.append(spacer*i - scaler)
        ticklist.append(spacer*i)
        ticklist.append(spacer*i + 0.2*scaler)
        ticklist.append(spacer * i + 0.4 * scaler)
    plt.ylim(0, 20)
    plt.ylabel('r [m]')
    plt.xlabel(r'$U/U_{0}$')
    plt.xticks(ticklist, ['0', '1', '1.2', '1.4', '0', '1', '1.2', '1.4', '0', '1', '1.2', '1.4'])
    # plt.legend(('EM', 'CFD'), loc='upper right')
    plt.legend(('thinCFD', 'thinEM', 'fatCFD', 'fatEM'), bbox_to_anchor=(0.75, 0.68))

    if plot == 1:
        plt.show()
        plt.close()
    elif plot == 2:
        plt.savefig(plot_path+plot_name, bbox_inches='tight')

def plot_rotor_grid(plot, wake, x_wake, y_wake, midline, rect_x, rect_y, nacelle_x, nacelle_y, x, y, U_local, V_local, Uinf, dx, dy, xlim, ylim):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    font = '24 '
    plt.rcParams.update({'font.size': font})
    plt.rcParams['font.size'] = font
    if type == 'rotor':
        plt.gca().set_aspect('equal', adjustable='box')
        plt.figure(1, figsize=(10, 10))
        if wake:
            plt.fill_between(x_wake, y_wake, where=y_wake >= midline, interpolate=True, color='blue')
        plt.fill_between(nacelle_x, nacelle_y, where=nacelle_y >= midline, interpolate=True, color='grey')
        plt.plot(rect_x, rect_y, color='k')
        # wake_bound = plt.plot(x_wake, y_wake, color='b')
        # wake_bound2 = plt.plot(x_wake, -y_wake, color='b', label='_nolegend_')
        plt.plot(x, y, color='k', linewidth=2)
        plt.quiver(x, y, U_local, V_local, angles='xy', scale_units='xy', scale=1,
                   color='r', label='Data', linewidths=100)
        plt.xlim((0, max(x) + Uinf))
        plt.ylim((0, max(y)))
        # plt.legend(('rotor', 'wake', 'nacelle', 'wind velocity'), loc='right')
        plt.tick_params(which='both')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        # plt.title('Wind Velocity [m/s]')
        if plot == 2:
            z = 1
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
        WS = WS / Uinf  # normalize for plot
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
        plt.plot(rect_x, rect_y, color='k')
        if wake:
            plt.plot(x_wake, y_wake, color='b')
        plt.colorbar(cf, fraction=0.0245, pad=0.02)
        plt.tick_params(which='both')
        plt.axis([-xlim, xlim, 0, ylim])
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.title('Wind Speed, '+'$\mathregular{U_{inf}}$'+' [m/s]')
        if plot == 2:
            zz = 1
            plt.savefig('_dataTmp/figures/contour_WS=' + str(int(Uinf)) + '.pdf', bbox_inches='tight')
        elif plot == 1:
            plt.show()
            plt.close()

            # FS = weio.read('_dataTmp/BEM_ws03_radial.csv')
            # FSdf = FS.toDataFrame()
            # NB = weio.read('_dataTmp/BEM_ws03_radial_blockage.csv')
            # NBdf = NB.toDataFrame()
            #
            # plt.rcParams['font.family'] = 'serif'
            # plt.rcParams['font.serif'] = ['Times New Roman']
            # font = '20 '
            # plt.rcParams.update({'font.size': font})
            # plt.rcParams['font.size'] = font
            # plt.plot(FSdf['r_[m]'], FSdf['WS [m/s]'])
            # plt.plot(NBdf['r_[m]'], NBdf['WS [m/s]'])
            # plt.xlabel('r (m)')
            # plt.ylabel('WS (m/s)')
            # plt.legend(('free stream', 'nacelle blockage'), loc='right')
            # plt.savefig('_dataTmp/figures/FS_vs_NB_WS_comparo.pdf', bbox_inches='tight')
            # plt.show()

# get_BLH('_dataTmp/turb/S2G1u_turb_CFD.npy')
# get_BLH('_dataTmp/turb/S2G2u_turb_CFD.npy')
# get_BLH('_dataTmp/turb/S3G1u_turb_CFD.npy')
# get_BLH('_dataTmp/turb/S3G2u_turb_CFD.npy')
# get_BLH('_dataTmp/turb/S3G3u_turb_CFD.npy')
# get_BLH('_dataTmp/turb/S3G4u_turb_CFD.npy')
# get_BLH('_dataTmp/turb/S3G5u_turb_CFD.npy')
# get_BLH('_dataTmp/turb/S4U1u_turb_CFD.npy')
# get_BLH('_dataTmp/turb/S4U2u_turb_CFD.npy')
# get_BLH('_dataTmp/turb/S4U3u_turb_CFD.npy')
plot_Re_stuff()
# compare_U()
# compare_geos()
# compare_heights_EM()
# compare_TI()
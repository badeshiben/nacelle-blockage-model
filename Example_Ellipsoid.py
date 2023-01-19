""" 
Computes the velocity about an ellipsoid of revolution

Origin is at the center of the ellipsoid
x is along the flow direction
r orthogonal to r. 

The flow is axisymmetric.

"""

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
# --- Local
from SourceEllipsoid import ser_u

if __name__=='__main__':

    # --- Ellipse parameters
    a  = 1.0     # Major axis of the ellipse (i.e. Nacelle length / 2)
    b  = 0.5   # Minor axis of the ellipse (i.e. Nacelle Width /2 )
    U0 = 10    # Free Stream


    # --- Velocity field on a radial line
    vxProbe = [-a, -a/2 ,0, a, a/2] # location on the x axis where we are probing
    vSty=['-','-','-','o','d']
    fig,ax = plt.subplots(1,1)
    for xProbe,sty in zip(vxProbe,vSty):
        vr = np.linspace(b,10*b,20)
        vx = vr*0  + xProbe  
        U,V   = ser_u(vx,vr,U0,a,b)
        bInEllipse=(vx**2/a**2+vr**2/b**2)<1
        U[bInEllipse]=np.nan
        ax.plot(vr/a,U/U0,sty,label='x/a={:.1f}'.format(xProbe/a))
    ax.set_xlabel('r/a [-]')
    ax.set_ylabel('Induced axial velocity/U0 [-]')
    ax.legend()


    # --- Velocity field on grid
    nx = 200
    nr = nx+1
    vx = np.linspace(-2*a,2*a,nx)
    vr = np.linspace(-4*b   ,4*b,nr)
    X,R = np.meshgrid(vx, vr)
    U,V   = ser_u(X,R,U0,a,b)
    # --- Plot
    Utot     = U+U0
    #Speed = np.sqrt((Utot**2+V**2))/U0
    Speed = Utot/U0
    bInEllipse=(X**2/a**2+R**2/b**2)<1
    Speed[bInEllipse]=np.nan
    fig,ax = plt.subplots(1,1)
    im = ax.contourf(X, R, Speed)
    cb=fig.colorbar(im)
    rseed=np.linspace(np.min(vr)*0.85,np.max(vr)*0.85,8)
    start=np.array([rseed*0-2*a*0.9,rseed])
    sp=ax.streamplot(vx,vr,Utot,V,color='k',start_points=start.T,linewidth=0.7,density=30,arrowstyle='-')
    ax.set_ylim([-4*b,4*b])
    ax.set_xlim([-2*a,2*a])
    ax.set_xlabel('x/a [-]')
    ax.set_ylabel('r/a [-]')
    ax.set_aspect('equal','box')
    ax.set_title('Source Ellipsoid Streamlines')

    plt.show()

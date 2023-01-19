
hline='\\hline\n'
# s += '- & CFD AbsAvgTurb & CFD dCP & CFD dCQ & CFD dCT	& CFD dEdge & CFD dFlap & EM/CFD AbsAvgTurb	& EM/CFD dCP'

def tolatex(M, linenames=None,fmt='{:5.2f}'):
    s = '\\begin{table}[!htbp]\n'
    s += '\scriptsize\n'
    s += '\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}\n'
    s += '\hline\n'
    s += '\centering\n'
    s += '$Re_{L}$ & Re_{H}$ & $\overline{\mid \Delta V \mid}_{CFD}$ & $\Delta CP_{CFD}$ & $\Delta CT_{CFD}$ & $\Delta Edge_{CFD}$ & $\Delta Flap_{CFD}$ & $E_{\Delta V}$ & $E_{CP}$'
    # s += '- & CFD Speedup & CFD dCP & CFD dCT & CFD dEdge & CFD dFlap'
    s += '\\\\\n'
    s += '\hline\n'
    for iline,line in enumerate(M):
        # if linenames is not None:
            # s+=linenames[iline]+' & '
        s+='\% & '.join([fmt.format(v) for v in line ])
        s+='\%\\\\\n'
        s+= '\hline\n'
    s += '\end{tabular}\n'
    s += '\caption{\label{tab:table-name}Flow/BEM results for Suite 4.}\n'
    s += '\end{table}'
    #s='\\\\\n'.join(['&'.join(["{:5.1f}".format(v) for v in line ]) for line in M]) 
    return s

if __name__=='__main__':
    import numpy as np
    import pandas as pd
    # M=np.zeros((2,3))
    # M[0,:]=1
    # M[1,:]=2
    # print(tolatex(M, linenames=['Sim1','Sim2']))
    # MTS2d = pd.read_excel('_dataTmp/mini_tables.xlsx', sheet_name='Suite4', header=0, nrows=3, index_col=0, usecols='A:G')
    MTS2d = pd.read_excel('_dataTmp/mini_tables.xlsx', sheet_name='Suite4', header=0, nrows=3, index_col=0, usecols='A, C:F, H:L')
    M = MTS2d.to_numpy()
    M *= 100
    ln = list(MTS2d.index)
    print(tolatex(M, linenames=ln))

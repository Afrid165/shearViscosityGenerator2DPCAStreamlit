import pickle

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@st.cache_resource
def loadPCA(pcaFile):
    with open(pcaFile, 'rb') as f:
        pca = pickle.load(f)
    return pca

def main():
    pcaFile = "shearPCA.pickle"
    pca = loadPCA(pcaFile)

    st.sidebar.header('Model Parameters:')
    params = []     # record the model parameter values
    for iPC in range(pca.n_components):
        parVal = st.sidebar.slider(label=f"PC: {iPC}",
                                   min_value=round(pca.pcMin[iPC], 2),
                                   max_value=round(pca.pcMax[iPC], 2),
                                   value=0.,
                                   step=(pca.pcMax[iPC] - pca.pcMin[iPC])/1000.,
                                   format='%f')
        
        params.append(parVal)
    params = np.array([params,])

    T_plot = np.linspace(0., 0.5, 10)
    mu_B_plot = np.linspace(0., 0.5, 10)
    T_values, mu_B_values = np.meshgrid(T_plot, mu_B_plot)
    #shear = pca.inverse_transform(params).flatten()
    shear = pca.inverse_transform(params).reshape(len(T_plot), len(mu_B_plot))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('T (GeV)')
    ax.set_ylabel(r"$\mu_B$ (GeV)")
    ax.set_zlabel(r"$\eta/s$")
    ax.plot_surface(T_values, mu_B_values, shear, cmap='viridis')
    st.pyplot(fig)
    '''
    plt.plot(T_plot, shear, '-r')
    plt.xlim([0, 0.5])
    plt.ylim([0, 0.5])
    plt.xlabel(r"T (GeV)")
    plt.ylabel(r"$\eta/s$")
    st.pyplot(fig)
    '''

if __name__ == '__main__':
    main()

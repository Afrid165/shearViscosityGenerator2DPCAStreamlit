from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PCATransformation import PCATransformation
from mpl_toolkits.mplot3d import Axes3D 

def transformation(eta_s):
    eta_s_max = 0.4
    scale = 1.0
    return eta_s_max/2.*(1. + np.tanh(scale*eta_s))

def eta_s_file_writer(T, mu_B, eta_s, filename):
    """
        This function writes the eta_s to a pickle file with a dictionary 
        for each eta_s. The different columns are: T, mu_B, eta_s
    """
    eta_s_dict = {}
    for es in range(len(eta_s)):
        #create a 2D grid of T and mu_B values
        T_grid, mu_B_grid = np.meshgrid(T, mu_B)
        #T_muB = np.column_stack((T_grid.ravel(), mu_B_grid.ravel()))
        
        #make a 3d array by stacking T_muB and eta_s[es]
        data = np.stack([T_grid, mu_B_grid, eta_s[es]], axis=2)
        print(f"Shape of data for eta_s {es:04}: {data.shape}")


        
     
        eta_s_dict[f'{es:04}'] = data
    with open(filename, 'wb') as f:
        pickle.dump(eta_s_dict, f)

def main(ranSeed: int, number_of_eta_s: int) -> None:
    # print out the minimum and maximum values of the training data x_values
    T_min = 0.0
    T_max = 0.5
    mu_B_min = 0.0
    mu_B_max = 0.5
   
    print(f"Minimum of the temperature is: {T_min}")
    print(f"Maximum of the temperature is: {T_max}")

    print(f"Minimum of the mu_B is: {mu_B_min}")
    print(f"Maximum of the mu_B is: {mu_B_max}")

    #create a grid of T and mu_B values
    T_values = np.linspace(T_min, T_max, 10)
    mu_B_values = np.linspace(mu_B_min, mu_B_max, 10)
    T_grid, mu_B_grid = np.meshgrid(T_values, mu_B_values, indexing='ij')
    T_muB = np.column_stack((T_grid.ravel(), mu_B_grid.ravel()))
    

    # print out the shape of T_muB
    print(f"Shape of T_muB: {T_muB.shape}")
            
 
    # set the random seed
    if ranSeed >= 0:
        randomness = np.random.seed(ranSeed)
    else:
        randomness = np.random.seed()


   
    correlation_length_min = 0.05
    correlation_length_max = 0.5

    eta_s_set = []
    nsamples_per_batch = max(1, int(number_of_eta_s/100))
    progress = 0
    while len(eta_s_set) < number_of_eta_s:
        correlation_length = np.random.uniform(correlation_length_min,
                                               correlation_length_max)
        print(f"Progress {progress}%, corr len = {correlation_length:.2f} ...")
        kernel = RBF(length_scale=correlation_length, length_scale_bounds="fixed")
        gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        print("Generating random functions...")
        eta_s_vs_T_GP = gpr.sample_y(T_muB, 
                                     n_samples=nsamples_per_batch, 
                                     random_state=randomness).transpose()
        print("Random functions generated")
        #print(f"Shape of eta_s_vs_T_GP: {eta_s_vs_T_GP.shape}")
        eta_s_vs_T_GP = transformation(eta_s_vs_T_GP)
        #eta_s_vs_T_GP = eta_s_vs_T_GP.transpose()
        for sample_i in eta_s_vs_T_GP:
            #eta_s_set.append(sample_i)
            print(f"Sample i shape: {sample_i.shape}")
            sample_i = sample_i.reshape(len(T_values), len(mu_B_values))  # reshape to match the 2D (T, mu_B) grid
            print(f"Sample i reshaped: {sample_i.shape}")
            eta_s_set.append(sample_i)
        progress += 1

    # make verification plots
    plt.figure().add_subplot(projection='3d')
    for i in range(number_of_eta_s):
        if i%(nsamples_per_batch*4) == 0:
            #plt.plot(T_plot, eta_s_set[i], '-')
            plt.contourf(T_values, mu_B_values, eta_s_set[i], levels=20, cmap='viridis', alpha=0.2)
            

    plt.xlim([T_min, T_max])
    plt.ylim([mu_B_min, mu_B_max])
    plt.xlabel(r"$T$ [GeV]")
    plt.ylabel(r"$\mu_B$ [GeV]")
    plt.colorbar(label=r"$\eta/s$")
    plt.title("Shear viscosity samples")
    plt.savefig("eta_s_samples_contour.png", dpi=600)
    plt.clf()

   
    # write the eta_s to a file
    eta_s_file_writer(T_values, mu_B_values, eta_s_set, f"eta_s.pkl")

    # check PCA
    plt.figure()
    varianceList = [0.9, 0.95, 0.99]
    for var_i in varianceList:
        scaler = StandardScaler()
        pca = PCA(n_components=var_i)

        eta_s_set = np.array(eta_s_set)
        print(f"Shape of eta_s_set: {eta_s_set.shape}")
        eta_s_set_2d = eta_s_set.reshape(eta_s_set.shape[0], -1)  # flatten the 3D data into a 2d one so that we can use Scaler and PCA
        scaled = scaler.fit_transform(eta_s_set_2d)
        PCA_fitted = pca.fit(scaled)
        print(f"Number of components = {pca.n_components_}")
        print(f"scaled shape: {scaled.shape}", scaled.shape)
        PCs = PCA_fitted.transform(scaled)
        # perform the inverse transform to get the original data
        zeta_s_reconstructed = PCA_fitted.inverse_transform(PCs)
        zeta_s_reconstructed = scaler.inverse_transform(zeta_s_reconstructed)
        zeta_s_reconstructed = zeta_s_reconstructed.reshape(eta_s_set.shape[0], eta_s_set.shape[1], eta_s_set.shape[2]) # reshape back to the original shape

        # calculate the RMS error between the original and reconstructed data

        RMS_error = np.sqrt(
            np.mean((eta_s_set - zeta_s_reconstructed)**2, axis=0))
        print(eta_s_set[1,2,3], zeta_s_reconstructed[1,2,3])
        
        plt.figure()
        ax = plt.subplot(111, projection='3d')
        ax.plot_surface(T_grid, mu_B_grid, RMS_error,
                 label=f"var = {var_i:.2f}, nPC = {pca.n_components_}")
        
    #plt.show()
    plt.legend()
    plt.savefig("RMS_errors_shear_viscosity.png")
    plt.clf()

    # check the distribution for PCs
    pca = PCATransformation(0.97)
    PCs = pca.fit_transform(eta_s_set_2d)
    print(f"Number of components = {PCs.shape[1]}")
    print(f"Number of samples = {PCs.shape[0]}")
    print(PCs.min(axis=0), PCs.max(axis=0))
    for i in range(PCs.shape[1]):
        plt.figure()
        plt.hist(PCs[:, i], bins=17, density=True)
        plt.savefig(f"PC{i}.png")
        plt.clf()

    with open("shearPCA.pickle", "wb") as f:
        pickle.dump(pca, f)

if __name__ == "__main__":
    ranSeed = 23
    number_of_eta_s = 100
    main(ranSeed, number_of_eta_s)


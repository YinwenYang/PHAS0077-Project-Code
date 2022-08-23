# Import some packages and Python modules we need for this project (i.e., UCLCHEM and SpectralRadex)
import uclchem
import spectralradex
from spectralradex import radex
import numpy as np
import pandas as pd
from multiprocessing import Pool
import os
import csv
from pandarallel import pandarallel

# Define a function to get the .dat file of the chemical species we are interested in
# These chemical species will be used in the RADEX model to produce molecular line intensities
def chemDat(i):
    switcher = {
        'C+': ['c+.dat'],
        'CH': ['ch-nohfs.dat'],
        'CH2': ['ch2_h2_ortho.dat', 'ch2_h2_para.dat'],
        'CH3CN': ['ch3cn.dat'],
        'CN': ['cn.dat'],
        'CO': ['co.dat'],
        'CS': ['cs@lique.dat'],
        'H2CO': ['oh2co-h2.dat', 'ph2co-h2.dat'],
        'H2O': ['oh2o@daniel.dat', 'ph2o@daniel.dat'],
        'H3O+': ['p-h3o+.dat', 'o-h3o+.dat'],
        'H2S': ['oh2s.dat', 'ph2s.dat'],
        'HCL': ['hcl.dat'],
        'HCN': ['hcn.dat'],
        'HCO+': ['hco+@xpol.dat'],
        'HCS+': ['hcs+@xpol.dat'],
        'HNC': ['hnc.dat'],
        'N2H+': ['n2h+@xpol.dat'],
        'NH3': ['p-nh3.dat', 'o-nh3.dat'],
        'NO': ['no.dat'],
        'OH': ['oh.dat'],
        'SIO': ['sio-h2.dat'],
        'SO': ['so@lique.dat']
    }
    return switcher.get(i, i + " is an invalid chemical given to chemDat")

# This part can be substituted with any choice of grids (i.e., combinations with different parameters)
# In this project, we vary: density, gas temperature and four elemental abundance (Carbon, Oxygen, Nitrogen, Sulfur) 
densities = np.logspace(3, 7, 30)
temperatures = np.linspace(10, 300, 30)
fc = np.logspace(-5, -3, 5)
fo = np.logspace(-5, -3, 5)
fn = np.logspace(-6, -4, 5)
fs = np.logspace(-7, -5, 5)

# Meshgrid will give all combinations, then we shape into columns and put into a table
parameterSpace = np.asarray(np.meshgrid(densities,temperatures,fc,fo,fn,fs)).T.reshape(-1, 6)
model_table=pd.DataFrame(parameterSpace.T, columns=['density','temperature','fc','fo','fn','fs'])

if os.path.exists("./UclchemToRadex_grid/Results.csv"):
    results = pd.read_csv("UclchemToRadex_grid/Results.csv", usecols=["Density", "gasTemp", "fc", "fo", "fn", "fs"])
    results = results.rename(columns={"Density":"density", "gasTemp":"temperature"})
    model_table = pd.concat([model_table, results, results]).drop_duplicates(keep=False)

# Define a function to connect the outputs of UCLCHEM model to the inputs of RADEX model
# First, run UCLCHEM model, extract variables that are required for RADEX and process the data (do some calculations)
# Because the original values from UCLCHEM cannot be directly used as the input values of RADEX model
def run_model(row):
    # Run UCLCHEM model and process the data
    PID = str(os.getpid())
    # basic settings of parameters we'll use for the grid
    ParameterDictionary = {"endatfinaldensity":False,
                           "freefall": False,
                           "initialDens": row.density,
                           "initialTemp": row.temperature,
                           "fc": row.fc,
                           "fo": row.fo,
                           "fn": row.fn,
                           "fs": row.fs,
                           "outputFile": PID + "temp.csv",
                           "finalTime":1.0e6,
                           "baseAv":2}
    result = uclchem.model.cloud(param_dict=ParameterDictionary)
    df = pd.read_csv(PID + "temp.csv", skiprows=2)
    col_names = df.columns.tolist()
    for index, value in enumerate(col_names):
        col_names[index] = value.replace(" ","")  # remove the space in the column names
    df.columns = col_names
    outspecies = "CO, CH3CN, CS, H2S, HCN, NH3, NO, SO"  # in this project, we only focus on these 8 molecular species
    columnsInterest = ['gasTemp','av','H','H+','H2'] + outspecies.split(", ")
    uclchem_data = df[columnsInterest]
    del df
    os.remove(PID + "temp.csv")
    # Do some calculations so that the data can be used as the RADEX inputs
    uclchem_data.insert(0,'Density','')
    uclchem_data['Density'] = row.density
    uclchem_data['H+'] = uclchem_data['H'] * uclchem_data['H+']
    uclchem_data['H2'] = uclchem_data['H'] * uclchem_data['H2']
    uclchem_data.insert(6,'e-','')
    uclchem_data['e-'] = uclchem_data['H'] * uclchem_data['H+']
    uclchem_data.insert(7,'fc','')
    uclchem_data['fc'] = row.fc
    uclchem_data.insert(8,'fo','')
    uclchem_data['fo'] = row.fo
    uclchem_data.insert(9,'fn','')
    uclchem_data['fn'] = row.fn
    uclchem_data.insert(10,'fs','')
    uclchem_data['fs'] = row.fs
    for chem in outspecies.split(", "):
        uclchem_data[chem] = uclchem_data[chem] * uclchem_data['av'] * 1.6e21
    # 1.6e21 is a constant in the formula of column density calculation
    uclchem_data = uclchem_data.iloc[[-1]]
    # Then, feed the outputs of UCLCHEM as the inputs into RADEX
    tkin = uclchem_data['gasTemp']
    nh = uclchem_data['H']
    nh2 = uclchem_data['H2']
    ne = uclchem_data['e-']
    nhx = uclchem_data['H+']
    for chem in outspecies.split(", "):
        molfile = chemDat(chem)
        cdmol = uclchem_data[chem]
        for file in molfile:
            results = spectralradex.radex.run_params(file, tkin, cdmol, nh, nh2, ne, nhx, fmin=200.0, fmax=800.0, output_file=None)
            for index, row in results.iterrows():
                name = chem + "_" + str(row["freq"]) + "_Flux(K*km/s)"
                Observation = row["FLUX (K*km/s)"]
                uclchem_data[name] = Observation
    # Keep track of where each model output will be saved and make sure that folder exists
    if not os.path.exists("./UclchemToRadex_grid"):
        os.makedirs("./UclchemToRadex_grid")
    if not os.path.isfile("./UclchemToRadex_grid/Results.csv"):
        uclchem_data.to_csv('./UclchemToRadex_grid/Results.csv', index = False)
    else:
        uclchem_data.to_csv('./UclchemToRadex_grid/Results.csv', mode='a', header=False, index = False)
    return

# Use the parallel in HPC to make the program run faster
pandarallel.initialize(nb_workers=32, use_memory_fs=False)
model_table.parallel_apply(run_model, axis=1)
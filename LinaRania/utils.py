import os
try :
  import comtypes.client
except:
  pass
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import itertools


def plot_scatter_by_z_levels(coord, selected_ids=None, points_per_row=4, figsize=(16, 10)):
    """
    Plot scatter subplots of X vs Y for each unique Z level.
    Hides axis ticks and minimizes subplot spacing.
    """
    coord = coord.astype(str)
    z_levels = np.unique(coord[:, 3].astype(float))
    n_levels = len(z_levels)
    rows = (n_levels + points_per_row - 1) // points_per_row

    fig, axes = plt.subplots(rows, points_per_row, figsize=figsize)
    axes = axes.flatten()

    for i, z in enumerate(z_levels):
        ax = axes[i]
        layer = coord[coord[:, 3].astype(float) == z]

        # Plot all points
        ax.scatter(layer[:, 1].astype(float), layer[:, 2].astype(float), s=5, color='gray')

        # Highlight selected sensors
        if selected_ids is not None:
            mask = np.isin(layer[:, 0], selected_ids)
            highlighted = layer[mask]
            ax.scatter(highlighted[:, 1].astype(float), highlighted[:, 2].astype(float),
                       s=20, color='red', edgecolors='black')

        # Title and clean axes
        ax.set_title(f"Level {i+1}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Remove spacing
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.show()

def run_batches(dataset_array, batch_size, SapModel, group_name, E_i, mat_names, not_damaged_file, output_dir):
    all_csv_paths = []
    n_batches = int(np.ceil(len(dataset_array) / batch_size))

    for i in range(n_batches):
        batch = dataset_array[i*batch_size:(i+1)*batch_size]
        print(f"\nProcessing batch {i+1}/{n_batches} with {len(batch)} samples...")

        # Generate dataset
        damage_indicators = create_dataset(batch, SapModel, group_name, E_i, mat_names, not_damaged_file)

        # Save batch to CSV
        batch_df = pd.DataFrame(damage_indicators)
        batch_path = os.path.join(output_dir, f"batch_{i+1:03d}.csv")
        batch_df.to_csv(batch_path, index=False)
        all_csv_paths.append(batch_path)

    return all_csv_paths


def compute_and_save_not_damaged(SapModel, group_name, E_i, mat_names, n_elements, out_path):
    print("Running the not damaged scenario...")
    Frequency_0, mode_shapes_0 = create_dp(SapModel, n_elements * [0], group_name, E_i, mat_names)
    notDamaged_Flexibility = FlexibilityMatrix(Frequency_0, mode_shapes_0)

    with open(out_path, "wb") as f:
        pickle.dump(notDamaged_Flexibility, f)
    print(f"✅ notDamaged_Flexibility saved to {out_path}")

def create_dataset(batch_array, SapModel, group_name, E_i, mat_names, not_damaged_path):
    # Load saved notDamaged_Flexibility
    with open(not_damaged_path, "rb") as f:
        notDamaged_Flexibility = pickle.load(f)

    DI = []
    for severity in tqdm(batch_array, desc='Generating batch'):
        Frequency, mode_shapes = create_dp(SapModel, severity, group_name, E_i, mat_names)
        Damaged_Flexibility = FlexibilityMatrix(Frequency, mode_shapes)
        Damage_indicator = DeltaFmax(Damaged_Flexibility, notDamaged_Flexibility)
        DI.append(Damage_indicator)

    return DI

def SetMaterial(mat_name, E, SapModel,):
    SapModel.PropMaterial.SetMPIsotropic(mat_name, E, 0.3, 0.00001)

def create_dp(SapModel, severity, group_name, E_i, mat_names):
    SapModel.SetModelIsLocked(False)
    # edit materials
    for i, s in enumerate(severity):
        E = E_i*(1-s)
        SetMaterial(mat_names[i], E, SapModel)
    # Run analysis
    SapModel.Analyze.RunAnalysis()
    # Get results
    SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
    SapModel.Results.Setup.SetCaseSelectedForOutput("MODAL")

    NumberResults = 0
    Obj = ""
    Elm = ""
    LoadCase = ""
    StepType = ""
    StepNum = []
    U1 = []
    U2 = []
    U3 = []
    R1 = []
    R2 = []
    R3 = []
    # Retrieve mode shape results
    NumberResults, Obj, Elm, LoadCase, StepType, StepNum, U1, U2, U3, R1, R2, R3, n = SapModel.Results.ModeShape(
        group_name, 2, NumberResults, Obj, Elm, LoadCase, StepType, StepNum, U1, U2, U3, R1, R2, R3
    )
    [_,_,_,_,_,_,Frequency,_,_] = SapModel.Results.ModalPeriod(0,'','',[],[],[],[],[])
    U1 = np.array(U1).reshape((12,-1))
    U2 = np.array(U2).reshape((12,-1))
    U3 = np.array(U3).reshape((12,-1))
    mode_shapes = np.concatenate([U1.T, U2.T, U3.T], axis = 0)

    return Frequency, mode_shapes

def FlexibilityMatrix(Frequency, mode_shapes):
    # Creating the Modal matrix
    f = 2 * np.pi * np.array(Frequency)
    f = np.diag(f)
    
    # Creating the flexibility matrix
    t = mode_shapes
    np.dot(f, t.T)
    Flexibility = np.dot(t, np.dot(f, t.T))
    return Flexibility
def DeltaFmax(Damaged_Flexibility, notDamaged_Flexibility):
    return np.max(np.abs(Damaged_Flexibility-notDamaged_Flexibility), axis = 1)

def generate_unique_combinations(n=5000, step=0.05, n_elements = 3):
    """
    Generate an (n x 3) array of unique rows,
    where each element is selected from np.arange(0, 0.9, step).

    Parameters:
        n (int): Number of unique rows to return.
        step (float): Step size for generating the value range.

    Returns:
        np.ndarray: Array of shape (n, 3) with unique rows.
    """
    values = np.arange(0, 0.9, step)
    all_combinations = np.array(list(itertools.product(values, repeat=n_elements)))

    if n > len(all_combinations):
        raise ValueError(f"Requested {n} combinations, but only {len(all_combinations)} are available.")

    np.random.shuffle(all_combinations)
    result = all_combinations[:n]
    return result


def start_API(verbose = True):
    '''
    Starts the API and creates an application to control it
    '''
    helper = comtypes.client.CreateObject('SAP2000v1.Helper')

    helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)

    mySapObject = helper.GetObject("CSI.SAP2000.API.SapObject") 
    try:
        mySapObject.ApplicationStart()
        if verbose:
            print("SAP2000 API started successfully")
    except : 
        if verbose:
            print("SAP2000 API failed to connect")
        return 1
    SapModel = mySapObject.SapModel

    return SapModel




import pandas as pd
from tqdm import tqdm
import pickle
import itertools

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_model(input_dim, layer_sizes, hidden_activation, output_dim, output_activation):
    input_layer = Input(shape=(input_dim,))
    x = input_layer
    for size in layer_sizes:
        x = Dense(size, activation=hidden_activation)(x)
    output = Dense(output_dim, activation=output_activation, name='output')(x)
    return Model(inputs=input_layer, outputs=output)

def plot_frames_and_rectangles_on_axis(ax, frames_coords, rectangles, zone1_value=None, zone2_value=None):
    green_red_cmap = mcolors.LinearSegmentedColormap.from_list('GreenRed', ['green', 'red'])
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Plot regular frames WITHOUT legend
    for frame in frames_coords:
        if frame['label'] == 'Frame':
            x = [frame['start'][0], frame['end'][0]]
            z = [frame['start'][1], frame['end'][1]]
            ax.plot(x, z, color='blue', alpha=0.7, linewidth=3)

    # Plot Zone 2 frame WITHOUT legend
    for frame in frames_coords:
        if frame['label'] == 'Zone':
            x = [frame['start'][0], frame['end'][0]]
            z = [frame['start'][1], frame['end'][1]]
            ax.plot(x, z, color='red', alpha=0.7, linewidth=3)
            
            # Calculate midpoint of the Zone 2 frame line
            mid_x = (x[0] + x[1]) / 2
            mid_z = (z[0] + z[1]) / 2
            
            # Add text near midpoint, shifted by +1000 on x axis
            ax.text(mid_x + 5000, mid_z, f"{zone2_value*100:.1f}%", color='black', fontsize=12,
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            break

    # Plot rectangles colored by zone values if provided, else default fill
    for key, rect in rectangles.items():
        if key == 'Zone' and zone1_value is not None:
            color = green_red_cmap(norm(zone1_value))
            ax.fill(rect['x'], rect['z'], color=color, alpha=0.6)
            # Calculate centroid of Zone 1 rectangle
            cx = np.mean(rect['x'])
            cz = np.mean(rect['z'])
            ax.text(cx, cz, f"{zone1_value*100:.1f}%", color='black', fontsize=12,
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        else:
            fill_color = 'grey' if key == 'Zone 1' else 'lightgrey'
            ax.fill(rect['x'], rect['z'], color=fill_color, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.grid(True)
    ax.axis('equal')

def plot_side_by_side(frames_coords, rectangles, pred_values, target_values):
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    zone1_pred, zone2_pred = pred_values
    zone1_target, zone2_target = target_values

    # Plot predicted
    plot_frames_and_rectangles_on_axis(axs[0], frames_coords, rectangles, zone1_pred, zone2_pred)
    axs[0].set_title('Predicted Values')

    # Plot target
    plot_frames_and_rectangles_on_axis(axs[1], frames_coords, rectangles, zone1_target, zone2_target)
    axs[1].set_title('Target Values')

    plt.tight_layout()
    plt.show()



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
    SapModel.Results.Setup.SetCaseSelectedForOutput("Modal")
    SapModel.Results.Setup.SetOptionModeShape(1,12,True)

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


def start_API_etabs(verbose = True):
    '''
    Starts the API and creates an application to control it
    '''
    helper = comtypes.client.CreateObject('ETABSv1.Helper')

    myETABSObject = helper.GetObject("CSI.ETABS.API.ETABSObject") 
    try:
        myETABSObject.ApplicationStart()
        if verbose:
            print("Etabs API started successfully")
    except : 
        if verbose:
            print("Etabs API failed to connect")
        return 1
    SapModel = myETABSObject.SapModel

    return SapModel
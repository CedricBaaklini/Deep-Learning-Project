import os
import torch
from torch_geometric.data import Data
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Paths and global settings
POSCAR_DIR = "/home/ah/Desktop/GNN/POSCAR"
CHARGE_DIR = "/home/ah/Desktop/GNN/CHARGE"
OUTPUT_DIR = "/home/ah/Desktop/GNN/Preprocessed_Data"
data_max_files = 500

# Create an output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Validate paths
print(f"POSCAR_DIR: {POSCAR_DIR}, Exists? {os.path.exists(POSCAR_DIR)}")
if os.path.exists(POSCAR_DIR):
    print(f"Sample files: {os.listdir(POSCAR_DIR)[:5]}")
print(f"CHARGE_DIR: {CHARGE_DIR}, Exists? {os.path.exists(CHARGE_DIR)}")
if os.path.exists(CHARGE_DIR):
    print(f"Sample files: {os.listdir(CHARGE_DIR)[:5]}")

# Expanded atomic properties
atomic_properties = {
    1: {"electronegativity": 2.20, "mass": 1.008, "valence": 1, "radius": 0.53},  # H
    3: {"electronegativity": 0.98, "mass": 6.941, "valence": 1, "radius": 1.52},  # Li
    6: {"electronegativity": 2.55, "mass": 12.011, "valence": 4, "radius": 0.67},  # C
    7: {"electronegativity": 3.04, "mass": 14.007, "valence": 5, "radius": 0.56},  # N
    8: {"electronegativity": 3.44, "mass": 15.999, "valence": 6, "radius": 0.48},  # O
    9: {"electronegativity": 3.98, "mass": 18.998, "valence": 7, "radius": 0.42},  # F
    11: {"electronegativity": 0.93, "mass": 22.990, "valence": 1, "radius": 1.90},  # Na
    14: {"electronegativity": 1.90, "mass": 28.085, "valence": 4, "radius": 1.11},  # Si
    15: {"electronegativity": 2.19, "mass": 30.974, "valence": 5, "radius": 0.98},  # P
    16: {"electronegativity": 2.58, "mass": 32.06, "valence": 6, "radius": 0.88},  # S
    17: {"electronegativity": 3.16, "mass": 35.45, "valence": 7, "radius": 0.79},  # Cl
    26: {"electronegativity": 1.83, "mass": 55.845, "valence": 2, "radius": 1.24},  # Fe
}

# Function to load and preprocess a single molecular graph
def preprocess_molecular_graph(poscar_path, charge_path):
    atoms = read(poscar_path, format='vasp')
    atomic_numbers = atoms.get_atomic_numbers()
    n_atoms = len(atoms)

    # Create node features
    x = torch.tensor([
        [
            atomic_numbers[i],
            atomic_properties.get(atomic_numbers[i], {"electronegativity": 0.0}).get("electronegativity", 0.0),
            atomic_properties.get(atomic_numbers[i], {"mass": 0.0}).get("mass", 0.0),
            atomic_properties.get(atomic_numbers[i], {"valence": 0.0}).get("valence", 0.0),
            atomic_properties.get(atomic_numbers[i], {"radius": 0.0}).get("radius", 0.0),
        ]
        for i in range(n_atoms)
    ], dtype=torch.float)

    # Compute pairwise distances with PBC
    distances = atoms.get_all_distances(mic=True)
    threshold = 5.0  # Increased threshold to ensure edges are created
    i, j = np.triu_indices(n_atoms, k=1)
    mask = distances[i, j] < threshold
    edge_pairs = np.stack([i[mask], j[mask]], axis=0)
    edge_distances = distances[i, j][mask]
    edge_index = np.concatenate([edge_pairs, edge_pairs[::-1]], axis=1)

    # Edge features
    edge_distances = np.concatenate([edge_distances, edge_distances])
    edge_attr_inv = 1.0 / (edge_distances + 1e-6)
    centers = np.linspace(0, threshold, 10)
    edge_attr_gauss = np.exp(-(edge_distances[:, None] - centers) ** 2 / 0.5)
    edge_attr = np.concatenate([edge_attr_inv[:, None], edge_attr_gauss], axis=1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Load charges
    with open(charge_path, 'r') as f:
        charges = [float(line.strip()) for line in f if line.strip()]
    charges = np.array(charges)

    if len(atoms) != len(charges):
        raise ValueError(f"Mismatch: {len(atoms)} atoms vs {len(charges)} charges in {poscar_path}")

    return x, edge_index, edge_attr, charges

# Function to preprocess the entire dataset
def preprocess_dataset(max_files):
    if not os.path.exists(POSCAR_DIR) or not os.path.exists(CHARGE_DIR):
        print("Directory not found!")
        return None, None, None

    poscar_files = sorted(os.listdir(POSCAR_DIR))[:min(max_files, len(os.listdir(POSCAR_DIR)))]
    all_charges = []
    valid_poscar_files = []
    preprocessed_data = []

    # Collect charges and validate files
    for poscar_file in poscar_files:
        try:
            number = poscar_file.split('_')[1].replace('.POSCAR', '')
            charge_path = os.path.join(CHARGE_DIR, f"CHARGE_{number}")
            if not os.path.exists(charge_path):
                print(f"Charge file not found for {poscar_file}. Skipping.")
                continue

            with open(charge_path, 'r') as f:
                charges = [float(line.strip()) for line in f if line.strip()]
            all_charges.extend(charges)
            valid_poscar_files.append(poscar_file)
        except Exception as e:
            print(f"Error processing {poscar_file}: {e}")
            continue

    if not all_charges:
        print("No valid charge data found. Exiting.")
        return None, None, None

    # Analyze charge distribution
    all_charges = np.array(all_charges)
    print(f"Global charge range: {np.min(all_charges):.4f} to {np.max(all_charges):.4f}")
    print(f"Mean charge: {np.mean(all_charges):.4f}, Std: {np.std(all_charges):.4f}")

    # Plot charge distribution
    plt.figure(figsize=(8, 6))
    plt.hist(all_charges, bins=50, density=True, alpha=0.7)
    plt.xlabel("Charge")
    plt.ylabel("Density")
    plt.title("Distribution of Atomic Charges")
    plt.savefig(os.path.join(OUTPUT_DIR, "charge_distribution.png"))
    plt.close()

    # Apply log-transform if needed
    charge_min, charge_max = np.min(all_charges), np.max(all_charges)
    if charge_min < 0:
        shifted_charges = all_charges - charge_min + 1e-6
    else:
        shifted_charges = all_charges + 1e-6
    log_charges = np.log(shifted_charges)
    skewness = (np.mean(log_charges) - np.median(log_charges)) / np.std(log_charges)
    print(f"Skewness of log-transformed charges: {skewness:.4f}")

    use_log_transform = abs(skewness) < 0.5
    if use_log_transform:
        print("Applying log-transform to charges.")
        charges_to_normalize = log_charges
        charge_min, charge_max = np.min(log_charges), np.max(log_charges)
        normalization_type = "log"
    else:
        print("Using standard normalization.")
        charges_to_normalize = all_charges
        charge_min, charge_max = np.min(all_charges), np.max(all_charges)
        normalization_type = "standard"

    # Normalize to [0, 1]
    charges_to_normalize = (charges_to_normalize - charge_min) / (charge_max - charge_min)
    print(f"Normalized charge range (should be 0 to 1): {np.min(charges_to_normalize):.4f} to {np.max(charges_to_normalize):.4f}")

    # Preprocess each graph
    charge_idx = 0
    for poscar_file in valid_poscar_files:
        try:
            poscar_path = os.path.join(POSCAR_DIR, poscar_file)
            number = poscar_file.split('_')[1].replace('.POSCAR', '')
            charge_path = os.path.join(CHARGE_DIR, f"CHARGE_{number}")
            x, edge_index, edge_attr, charges = preprocess_molecular_graph(poscar_path, charge_path)

            # Normalize charges
            if use_log_transform:
                if np.min(charges) < 0:
                    charges = charges - np.min(charges) + 1e-6
                else:
                    charges = charges + 1e-6
                charges = np.log(charges)
            charges = (charges - charge_min) / (charge_max - charge_min)
            y = torch.tensor(charges, dtype=torch.float).unsqueeze(1)

            # Create Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            preprocessed_data.append(data)
        except Exception as e:
            print(f"Error preprocessing {poscar_file}: {e}")
            continue

    # Save preprocessed data
    with open(os.path.join(OUTPUT_DIR, "preprocessed_dataset.pkl"), 'wb') as f:
        pickle.dump(preprocessed_data, f)
    with open(os.path.join(OUTPUT_DIR, "normalization_params.pkl"), 'wb') as f:
        pickle.dump({
            "charge_min": charge_min,
            "charge_max": charge_max,
            "normalization_type": normalization_type
        }, f)

    return preprocessed_data, charge_min, charge_max

# Main execution
def main():
    preprocessed_data, charge_min, charge_max = preprocess_dataset(max_files=data_max_files)
    if preprocessed_data is None or len(preprocessed_data) == 0:
        print("No data preprocessed. Exiting.")
        return

    print(f"Preprocessed {len(preprocessed_data)} graphs.")
    print(f"Charge normalization range: {charge_min:.4f} to {charge_max:.4f}")
    print(f"Sample graph: {preprocessed_data[0]}")

if __name__ == "__main__":
    main()

from rdkit import Chem
from rdkit.Chem import AllChem
from pygamma import spin_system, Hcs, Fm, gen_op
import numpy as np

def generate_rmn_spectrum(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Impossible de générer la molécule à partir de SMILES.")
        return

    # Génération du spectre RMN
    AllChem.Compute2DCoords(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    num_atoms = mol.GetNumAtoms()
    spinsys = spin_system(num_atoms)
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        spinsys.set_zeeman(2 * np.pi * 200 * atom.GetIsotope().GetAtomicNum(), i)
    H0 = Hcs(spinsys)
    F = Fm(spinsys)
    H = H0 + F
    E, V = H.diagonalize()
    prediction = np.sort(E)
    return prediction

# Exemple d'utilisation
if __name__ == "__main__":
    molecule_smiles = input("Entrez la molécule (en SMILES) : ")
    spectrum = generate_rmn_spectrum(molecule_smiles)
    print("Prédiction du spectre RMN (en Hz) :")
    print(spectrum)

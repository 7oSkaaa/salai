import streamlit as st
import torch
import pandas as pd
import os
import time
import tempfile
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser, PPBuilder
import py3Dmol

from GraphDTA.models.ginconv import GINConvNet
from GraphDTA.inference_utils import smile_to_graph, seq_cat
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

# Setup
ROOT_DIR = Path(__file__).parent
MODEL_PATH = ROOT_DIR / "model_GINConvNet_combined.model"
KIBA_DATA_PATH = ROOT_DIR / "data" / "kiba_test.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GINConvNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

st.set_page_config(page_title="Protein-Ligand Interaction Prediction", layout="centered")
st.title("🤜 Predict Protein-Ligand Interaction (GraphDTA)")

uploaded_protein = st.file_uploader("🔬 Upload a PDB file", type=["pdb"])
predict_button = st.button("🔍 Predict Top 5 Ligands")

def extract_sequence_from_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb_path)
    ppb = PPBuilder()
    sequence = ""
    for pp in ppb.build_peptides(structure):
        sequence += str(pp.get_sequence())

    # Limit sequence length to 1000 to avoid potential issues
    if len(sequence) > 1000:
        sequence = sequence[:1000]
        st.warning("Protein sequence truncated to 1000 amino acids for processing.")

    return sequence

if predict_button and uploaded_protein:
    start = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        protein_pdb = tmp_path / "protein.pdb"
        protein_pdb.write_bytes(uploaded_protein.getvalue())
        sequence = extract_sequence_from_pdb(str(protein_pdb))

        if not sequence:
            st.error("Could not extract protein sequence from PDB.")
            st.stop()

        protein_tensor = torch.tensor(seq_cat(sequence), dtype=torch.long).to(device)

        # Ensure protein sequence is exactly 1000 amino acids as expected by the model
        if protein_tensor.size(0) < 1000:
            # Pad with zeros if shorter
            padded = torch.zeros(1000, dtype=torch.long, device=device)
            padded[:protein_tensor.size(0)] = protein_tensor
            protein_tensor = padded
        else:
            # Truncate if longer
            protein_tensor = protein_tensor[:1000]

        df = pd.read_csv(KIBA_DATA_PATH)
        smiles_col = next((col for col in df.columns if 'smiles' in col.lower()), None)
        if not smiles_col:
            st.error("SMILES column not found.")
            st.stop()

        smiles_list = df[smiles_col].tolist()[:1000]
        data_list, smiles_valid = [], []

        for smile in smiles_list:
            try:
                c_size, features, edge_index = smile_to_graph(smile)

                # Skip invalid molecules (which return None values)
                if c_size is None or features is None or edge_index is None:
                    continue

                x = torch.tensor(features, dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                # Fixed batch assignment to match the number of nodes in the graph
                batch = torch.zeros(x.size(0), dtype=torch.long)
                # Create a data object with the correct target format
                data = Data(x=x, edge_index=edge_index, batch=batch)
                # Set the target as a separate attribute and unsqueeze it for the expected shape
                data.target = protein_tensor.unsqueeze(0)  # Model expects shape [1, 1000]
                data_list.append(data)
                smiles_valid.append(smile)
            except Exception as e:
                # Silent exception handling to avoid cluttering the UI
                continue

        if not data_list:
            st.error("No valid ligand graphs.")
            st.stop()

        # Process in smaller batches to avoid memory issues
        loader = DataLoader(data_list, batch_size=32)
        predictions = []

        with st.spinner("Running predictions..."):
            success_count = 0
            for idx, batch in enumerate(loader):
                try:
                    batch = batch.to(device)
                    with torch.no_grad():
                        output = model(batch)
                        predictions.extend(output.cpu().numpy().flatten())
                    success_count += len(batch)
                    # Update progress
                    if idx % 5 == 0:
                        st.write(f"Processed {success_count}/{len(data_list)} ligands...")
                except Exception as e:
                    st.warning(f"Skipped some predictions due to compatibility issues.")
                    continue

        results = sorted(zip(smiles_valid, predictions), key=lambda x: x[1], reverse=True)[:5]

        st.subheader("🔝 Top 5 Predicted Ligands")
        st.dataframe(pd.DataFrame(results, columns=["SMILES", "Score"]))

        for i, (smile, score) in enumerate(results, 1):
            mol = Chem.MolFromSmiles(smile)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
            ligand_block = Chem.MolToPDBBlock(mol)

            st.markdown(f"### 🧪 Ligand #{i} — Score: {float(score):.4f}")

            # Create a viewer that shows both protein and ligand
            viewer = py3Dmol.view(width=600, height=400)

            # Add protein
            with open(str(protein_pdb), 'r') as f:
                protein_data = f.read()
            viewer.addModel(protein_data, 'pdb')
            viewer.setStyle({'model': -1}, {'cartoon': {'color': 'lightblue'}})

            # Add ligand
            viewer.addModel(ligand_block, 'pdb')
            viewer.setStyle({'model': 0}, {'stick': {'colorscheme': 'cyanCarbon', 'radius': 0.2}})

            # Configure viewer
            viewer.setBackgroundColor('white')
            viewer.zoomTo()
            viewer.spin(True)  # Enable spinning for better visualization

            # Show combined visualization
            st.components.v1.html(viewer._make_html(), height=420)

            # Add download buttons for the structures
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label=f"Download Ligand #{i} PDB",
                    data=ligand_block,
                    file_name=f"ligand_{i}.pdb",
                    mime="chemical/x-pdb"
                )
            with col2:
                st.download_button(
                    label="Download Protein PDB",
                    data=protein_data,
                    file_name="protein.pdb",
                    mime="chemical/x-pdb"
                )

        st.success(f"✅ Completed in {time.time() - start:.2f} seconds.")

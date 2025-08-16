# Salai - Protein-Ligand Interaction Prediction

An advanced protein-ligand interaction prediction tool using GraphDTA (Graph Deep Learning for Drug-Target Affinity) with comprehensive chemical database integration.

## Features

### 🧪 Protein-Ligand Interaction Prediction
- Upload protein structures (PDB format)
- AI-powered ligand ranking using GraphDTA
- 3D molecular visualization with py3Dmol
- Docking score calculation
- Interactive protein-ligand complex visualization

### 🧬 Protein-Protein Interaction Prediction
- Predict protein-protein interactions
- 3D complex visualization
- Download PDB files of predicted complexes

### 🔍 Chemical Database Integration
- **DrugBank API**: Official API with comprehensive drug data (requires subscription)
- **FDA Orange Book**: Free US FDA approved drugs database
- **ChEMBL**: Free European medicines database
- **PubChem**: Chemical structure and property database
- **Automatic fallback system** for maximum reliability

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/7oSkaaa/salai.git
   cd salai
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional - Set up API keys:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys (optional)
   ```

## Usage

### Running the Application

```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501`

### Chemical Database Configuration

The application works out-of-the-box with free APIs. For enhanced functionality:

1. **DrugBank API** (optional, requires subscription):
   - Get API key from [DrugBank](https://go.drugbank.com/releases/latest#open-data)
   - Add to `.env`: `DRUGBANK_API_KEY=your_key_here`

2. **Free APIs** (default, no setup required):
   - FDA Orange Book API
   - ChEMBL API
   - PubChem API

### Testing the Setup

```bash
python test_drugbank.py
```

This will test all chemical database integrations and show which APIs are working.

## Data Sources

### Ligand Data Sources Available:
1. **DrugBank (Approved Drugs)** - FDA/EMA approved medications
2. **PubChem** - Comprehensive chemical database
3. **ChEMBL** - Bioactivity database
4. **Local CSV** - Upload your own dataset

### Automatic Fallback System:
1. DrugBank API (if key provided)
2. FDA Orange Book + PubChem
3. ChEMBL approved drugs  
4. Curated static dataset

## Architecture

### Core Components:
- **GraphDTA Model**: Deep learning model for drug-target affinity prediction
- **Chemical Databases**: Multi-source chemical compound integration
- **Protein Interaction**: Protein-protein interaction prediction
- **3D Visualization**: Interactive molecular visualization

### Key Files:
- `main.py` - Main Streamlit application
- `chemical_databases.py` - Database integration layer
- `protein_interaction.py` - Protein interaction prediction
- `GraphDTA/` - GraphDTA model implementation

## API Integration Details

### DrugBank API Features:
- Real-time approved drug data
- Comprehensive drug information
- SMILES structures
- Drug indications and descriptions

### Free API Features:
- FDA Orange Book integration
- ChEMBL approved drug filtering
- PubChem SMILES lookup
- Automatic error handling and fallbacks

## Visualization Features

### Protein Visualization:
- Multiple rendering styles (Cartoon, Stick, Line, Cross, Sphere)
- Color schemes (Chain, Spectrum, Secondary Structure, B-factor)
- Interactive controls (rotation, zoom, spin)
- Surface representation with transparency

### Ligand-Protein Complex:
- Proper ligand positioning near protein center
- Realistic binding visualization
- Docking score calculation
- Downloadable PDB files

## System Requirements

- Python 3.8+
- RDKit (for chemical structure processing)
- PyTorch (for GraphDTA model)
- BioPython (for protein structure handling)
- Streamlit (for web interface)

## Troubleshooting

### RDKit Installation Issues:
```bash
# Install via conda (recommended)
conda install -c conda-forge rdkit

# Or check installation
python check_rdkit.py
```

### API Connection Issues:
- Check internet connection
- APIs may have temporary outages
- Application automatically falls back to alternatives

### Performance:
- Large protein files may take longer to process
- Reduce max_compounds parameter for faster results
- Use smaller batch sizes for memory-constrained systems

## Development

### Adding New Chemical Databases:
1. Implement new functions in `chemical_databases.py`
2. Add to the fallback chain in `get_drugbank_approved_drugs()`
3. Update the data source selector in `main.py`

### Extending Visualization:
- Modify py3Dmol settings in the visualization sections
- Add new color schemes or rendering styles
- Implement additional molecular representations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```
Salai: Protein-Ligand Interaction Prediction Tool
GitHub: https://github.com/7oSkaaa/salai
```

## Support

- **Documentation**: See `DRUGBANK_SETUP.md` for detailed API setup
- **Issues**: Report bugs on GitHub Issues
- **Features**: Request features via GitHub Issues

## Acknowledgments

- GraphDTA model implementation
- RDKit for chemical informatics
- BioPython for protein structure handling
- Streamlit for the web interface
- ChEMBL, PubChem, and FDA for free chemical databases

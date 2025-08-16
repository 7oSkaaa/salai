# Available Data Sources

This project now supports the following chemical compound data sources:

## 1. PubChem
- **Description**: NCBI's public repository of chemical information
- **Usage**: Search by compound name, SMILES, or chemical formula
- **Example searches**: "aspirin", "ibuprofen", "acetaminophen"
- **API**: Free public API with no authentication required
- **Features**: 
  - Real-time compound search
  - SMILES, molecular formulas, and compound names
  - Large database of chemical compounds

## 2. ChEMBL
- **Description**: European Bioinformatics Institute's database of bioactive compounds
- **Usage**: Search by compound name, SMILES, or activity data
- **Example searches**: "aspirin", drug names, target proteins
- **API**: Free public API with no authentication required
- **Features**:
  - Bioactivity data
  - Drug-like compounds
  - Target protein information
  - Clinical trial status

## 3. Local KIBA Dataset
- **Description**: Kinase Inhibitor BioActivity dataset
- **Usage**: Pre-loaded dataset for testing and demonstration
- **Size**: ~19,000 compounds with binding affinity data
- **Features**:
  - Fast local access
  - No internet connection required
  - Protein-compound interaction data
  - Binding affinity scores

## Search Tips

### For PubChem and ChEMBL:
- Use common drug names (e.g., "aspirin", "ibuprofen")
- Search by indication (e.g., "antibiotic", "antidepressant")
- Use SMILES strings for exact molecular searches
- Start with broad terms and refine as needed

### Search Examples:
- **By drug name**: "aspirin", "metformin", "atorvastatin"
- **By indication**: "analgesic", "antibiotic", "antiviral"
- **By SMILES**: "CC(=O)OC1=CC=CC=C1C(=O)O" (aspirin)

## Data Quality
- All sources provide validated SMILES strings
- Duplicate compounds are automatically removed
- Invalid SMILES are filtered out during processing
- Fallback to default compounds if searches fail

## Performance Notes
- PubChem: ~1-2 seconds per search
- ChEMBL: ~2-3 seconds per search  
- KIBA: Instant (local data)
- Batch processing supported for all sources

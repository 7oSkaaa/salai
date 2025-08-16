#!/usr/bin/env python3
"""
Test script for the chemical database integrations (PubChem and ChEMBL only)
"""

import sys
import pandas as pd
from chemical_databases import search_pubchem, search_chembl

def test_pubchem():
    """Test PubChem search functionality"""
    print("=== Testing PubChem ===")
    try:
        # Test with a common drug
        df = search_pubchem("aspirin", 5)
        if not df.empty and 'compound_iso_smiles' in df.columns:
            print(f"✅ PubChem: Found {len(df)} compounds")
            print(f"   First compound: {df.iloc[0]['name'] if 'name' in df.columns else 'Unknown'}")
            return True
        else:
            print("❌ PubChem: No valid compounds found")
            return False
    except Exception as e:
        print(f"❌ PubChem error: {str(e)}")
        return False

def test_chembl():
    """Test ChEMBL search functionality"""
    print("=== Testing ChEMBL ===")
    try:
        # Test with a common drug
        df = search_chembl("aspirin", 5)
        if not df.empty and 'compound_iso_smiles' in df.columns:
            print(f"✅ ChEMBL: Found {len(df)} compounds")
            print(f"   First compound: {df.iloc[0]['name'] if 'name' in df.columns else 'Unknown'}")
            return True
        else:
            print("❌ ChEMBL: No valid compounds found")
            return False
    except Exception as e:
        print(f"❌ ChEMBL error: {str(e)}")
        return False

def test_kiba_data():
    """Test local KIBA data loading"""
    print("=== Testing KIBA Data ===")
    try:
        df = pd.read_csv("data/kiba_test.csv")
        if not df.empty:
            smiles_col = next((col for col in df.columns if 'smiles' in col.lower()), None)
            if smiles_col:
                print(f"✅ KIBA: Found {len(df)} compounds with SMILES column '{smiles_col}'")
                return True
            else:
                print("❌ KIBA: No SMILES column found")
                return False
        else:
            print("❌ KIBA: No data found")
            return False
    except Exception as e:
        print(f"❌ KIBA error: {str(e)}")
        return False

def main():
    """Run all database tests"""
    print("🧪 Testing Chemical Database Integrations")
    print("=" * 50)
    
    results = []
    results.append(test_pubchem())
    results.append(test_chembl())
    results.append(test_kiba_data())
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   PubChem: {'✅ PASS' if results[0] else '❌ FAIL'}")
    print(f"   ChEMBL:  {'✅ PASS' if results[1] else '❌ FAIL'}")
    print(f"   KIBA:    {'✅ PASS' if results[2] else '❌ FAIL'}")
    
    if all(results):
        print("\n🎉 All database integrations are working correctly!")
        return True
    else:
        print(f"\n⚠️  Some tests failed. {sum(results)}/{len(results)} passed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

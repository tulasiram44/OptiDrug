"""
OptiDrug - Synthetic Demo Dataset Generator and Target Knowledgebase
Provides biologically grounded pipeline metadata (Disease -> Gene -> Protein -> Drug Target -> Compounds)
and calibrated synthetic candidate compound datasets with realistic molecular descriptors.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Biologically Grounded Target Catalog
TARGET_CATALOG: Dict[str, Dict[str, Any]] = {
    "EGFR": {
        "target_id": "EGFR",
        "disease": "Non-Small Cell Lung Cancer (NSCLC)",
        "gene_symbol": "EGFR",
        "ncbi_gene_id": "1956",
        "uniprot_id": "P00533",
        "protein_name": "Epidermal Growth Factor Receptor (ErbB-1 / HER1)",
        "drug_target_pocket": "Intracellular Tyrosine Kinase Domain (ATP-binding cleft, Met793 / Thr790)",
        "biological_mechanism": "Inhibition of receptor autophosphorylation and downstream MAPK / PI3K-AKT survival cascades",
        "benchmark_drugs": ["Gefitinib", "Erlotinib", "Osimertinib", "Afatinib"],
        "scaffolds": ["Quinazoline", "Anilinoquinazoline", "Pyrazolopyrimidine", "Indole-carboxamide"],
        "mw_range": (340.0, 530.0),
        "logp_range": (2.2, 5.2),
        "tpsa_range": (55.0, 120.0),
        "ideal_properties": {
            "mw": 446.9,
            "logp": 3.6,
            "hbd": 2,
            "hba": 6,
            "tpsa": 68.0,
            "rotatable_bonds": 6
        },
        "dna_simulation": {
            "gene_locus": "Chr 7p11.2 (Exon 21)",
            "mutation_name": "c.2573T>G (p.L858R)",
            "nucleotide_pos": 2573,
            "wildtype_dna": "ATGCGATTCCTGAGCGTACTG",
            "mutant_dna": "ATGCGATTCCGGAGCGTACTG",
            "mutation_index": 10,
            "wt_nt": "T",
            "mut_nt": "G",
            "wt_codon": "CTG (Leucine / Leu)",
            "mut_codon": "CGG (Arginine / Arg)",
            "chemical_change": "Hydrophobic nonpolar residue substituted by bulky, positively charged basic residue",
            "structural_impact": "Disrupts auto-inhibitory helical packing in the N-lobe, permanently locking the kinase activation loop into an open, constitutively hyperactive conformation independent of EGF binding.",
            "cellular_consequence": "Unchecked receptor autophosphorylation drives hyper-activation of RAS-RAF-MEK-ERK and PI3K-AKT cell survival pathways, driving rapid non-small cell lung tumorigenesis.",
            "drug_rescue": "High-affinity candidate TKI binds the mutant kinase ATP pocket, forming hinge H-bonds with Met793 and hydrophobic contacts with Leu718, extinguishing phosphorylation."
        }
    },
    "BRAF": {
        "target_id": "BRAF",
        "disease": "Melanoma & Colorectal Cancer",
        "gene_symbol": "BRAF",
        "ncbi_gene_id": "673",
        "uniprot_id": "P15056",
        "protein_name": "Serine/threonine-protein kinase B-raf (V600E mutant)",
        "drug_target_pocket": "Kinase catalytic cleft / DFG-out conformation pocket (Lys483, Glu501, Asp594)",
        "biological_mechanism": "Selective ATP-competitive inhibition of hyperactive BRAF V600E monomer/dimer signalling",
        "benchmark_drugs": ["Vemurafenib", "Dabrafenib", "Encorafenib"],
        "scaffolds": ["Pyrrolo[2,3-b]pyridine", "Thiazole-sulfonamide", "Pyridine-carboxamide", "Imidazothiazole"],
        "mw_range": (380.0, 560.0),
        "logp_range": (2.5, 5.5),
        "tpsa_range": (70.0, 135.0),
        "ideal_properties": {
            "mw": 489.9,
            "logp": 3.9,
            "hbd": 2,
            "hba": 7,
            "tpsa": 88.0,
            "rotatable_bonds": 5
        },
        "dna_simulation": {
            "gene_locus": "Chr 7q34 (Exon 15)",
            "mutation_name": "c.1799T>A (p.V600E)",
            "nucleotide_pos": 1799,
            "wildtype_dna": "CTAGCTACAGTGAAATCTCGAT",
            "mutant_dna": "CTAGCTACAGAGAAATCTCGAT",
            "mutation_index": 10,
            "wt_nt": "T",
            "mut_nt": "A",
            "wt_codon": "GTG (Valine / Val)",
            "mut_codon": "GAG (Glutamic Acid / Glu)",
            "chemical_change": "Hydrophobic nonpolar valine substituted by negatively charged carboxylic acid side chain",
            "structural_impact": "The glutamate negative charge phosphomimics physiological activation, electrostatically breaking the hydrophobic auto-inhibitory interaction between Gly464 and Val600.",
            "cellular_consequence": "Monomeric kinase activation reaches 500-fold above wild-type basal levels, continuously hyperphosphorylating MEK1/2 and driving malignant melanocyte proliferation.",
            "drug_rescue": "Candidate RAF inhibitor docks into the DFG-out pocket, locking the activation segment and abolishing downstream phosphorylation."
        }
    },
    "BACE1": {
        "target_id": "BACE1",
        "disease": "Alzheimer's Disease",
        "gene_symbol": "BACE1",
        "ncbi_gene_id": "23621",
        "uniprot_id": "P56817",
        "protein_name": "Beta-secretase 1 (Aspartyl protease)",
        "drug_target_pocket": "Catalytic dyad pocket (Asp32 and Asp228) with flap region (Val67-Glu77)",
        "biological_mechanism": "Inhibition of rate-limiting cleavage of amyloid precursor protein (APP) into neurotoxic Aβ peptides",
        "benchmark_drugs": ["Verubecestat", "Atabecestat", "Lanabecestat"],
        "scaffolds": ["Iminothiazine", "Amino-oxazoline", "Bicyclic-hydantoin", "Tetrahydropyrimidine"],
        "mw_range": (320.0, 490.0),
        "logp_range": (1.4, 3.8),
        "tpsa_range": (60.0, 115.0),
        "ideal_properties": {
            "mw": 410.0,
            "logp": 2.6,
            "hbd": 2,
            "hba": 5,
            "tpsa": 76.0,
            "rotatable_bonds": 4
        },
        "dna_simulation": {
            "gene_locus": "Chr 11q23.3 (Catalytic Dyad)",
            "mutation_name": "c.2009G>A (Swedish APP-BACE1 hyper-cleavage locus)",
            "nucleotide_pos": 2009,
            "wildtype_dna": "GAAGTGAAATTCGATGCCGAA",
            "mutant_dna": "GAAGTGAAATTAGATGCCGAA",
            "mutation_index": 12,
            "wt_nt": "C",
            "mut_nt": "A",
            "wt_codon": "TCG (Serine / Ser)",
            "mut_codon": "TAG (Stop/Variant)",
            "chemical_change": "Alters steric geometry of the catalytic cleft flap region (Val67-Glu77)",
            "structural_impact": "Lowers the catalytic activation energy of the Asp32/Asp228 dyad for amyloid precursor protein hydrolysis.",
            "cellular_consequence": "Accelerates toxic Aβ42 peptide generation, inducing rapid oligomerization, neurofibrillary tangle cascades, and synaptic loss.",
            "drug_rescue": "Small-molecule aspartyl protease inhibitor forms bidirectional H-bonds with Asp32/Asp228, freezing catalytic flap movement."
        }
    },
    "DPP4": {
        "target_id": "DPP4",
        "disease": "Type 2 Diabetes Mellitus",
        "gene_symbol": "DPP4",
        "ncbi_gene_id": "1803",
        "uniprot_id": "P27487",
        "protein_name": "Dipeptidyl Peptidase 4 (CD26)",
        "drug_target_pocket": "S1 and S2 subsite pockets (catalytic Ser630, Asp708, His740 triad)",
        "biological_mechanism": "Inhibition of rapid degradation of incretin hormones GLP-1 and GIP to stimulate glucose-dependent insulin secretion",
        "benchmark_drugs": ["Sitagliptin", "Saxagliptin", "Linagliptin", "Alogliptin"],
        "scaffolds": ["Triazolopiperazine", "Xanthine-core", "Pyrrolidine-nitrile", "Piperidinedione"],
        "mw_range": (310.0, 480.0),
        "logp_range": (0.8, 3.5),
        "tpsa_range": (65.0, 125.0),
        "ideal_properties": {
            "mw": 407.3,
            "logp": 2.1,
            "hbd": 2,
            "hba": 6,
            "tpsa": 77.0,
            "rotatable_bonds": 5
        },
        "dna_simulation": {
            "gene_locus": "Chr 2q24.2 (Catalytic Domain)",
            "mutation_name": "c.1889G>A (p.R630H Variant Region)",
            "nucleotide_pos": 1889,
            "wildtype_dna": "TACAGCTACGGCAGCGGTATC",
            "mutant_dna": "TACAGCTACAACAGCGGTATC",
            "mutation_index": 10,
            "wt_nt": "G",
            "mut_nt": "A",
            "wt_codon": "GGC (Glycine / Gly)",
            "mut_codon": "AAC (Asparagine / Asn)",
            "chemical_change": "Flexible nonpolar glycine replaced with polar asparagine near the Ser630 triad",
            "structural_impact": "Modulates S1 hydrophobic subsite conformation, accelerating the catalytic cleavage of the N-terminal His7-Ala8 dipeptide of GLP-1.",
            "cellular_consequence": "Incretin GLP-1 plasma half-life drops below 2 minutes, impairing glucose-dependent insulin secretion from pancreatic beta cells.",
            "drug_rescue": "Competitive DPP-4 candidate locks into S1/S2 pockets, establishing ionic salt-bridges that shield GLP-1 from hydrolysis."
        }
    },
    "ERBB2": {
        "target_id": "ERBB2",
        "disease": "HER2-Positive Breast Cancer",
        "gene_symbol": "ERBB2",
        "ncbi_gene_id": "2064",
        "uniprot_id": "P04626",
        "protein_name": "Receptor tyrosine-protein kinase erbB-2 (HER2 / NEU)",
        "drug_target_pocket": "Kinase domain ATP binding site (Cys805 covalent anchor site, Lys753)",
        "biological_mechanism": "Dual inhibition of HER2 and EGFR tyrosine kinase phosphorylation to arrest tumor cell proliferation",
        "benchmark_drugs": ["Lapatinib", "Neratinib", "Tucatinib", "Pyrotinib"],
        "scaffolds": ["Furanylquinazoline", "Anilinopyrimidine", "Pyrazolo-triazine", "Benzodioxole-amine"],
        "mw_range": (390.0, 580.0),
        "logp_range": (2.8, 5.4),
        "tpsa_range": (60.0, 130.0),
        "ideal_properties": {
            "mw": 510.5,
            "logp": 4.1,
            "hbd": 2,
            "hba": 7,
            "tpsa": 84.0,
            "rotatable_bonds": 7
        },
        "dna_simulation": {
            "gene_locus": "Chr 17q12 (Exon 20 insertion)",
            "mutation_name": "c.2313_2324dup (p.Y772_A775dup)",
            "nucleotide_pos": 2313,
            "wildtype_dna": "TACGTCATGGCCAGCGTGGA",
            "mutant_dna": "TACGTCATGATGATGAGCGTGGA",
            "mutation_index": 11,
            "wt_nt": "G",
            "mut_nt": "A",
            "wt_codon": "GCC (Alanine / Ala)",
            "mut_codon": "ATG (Methionine Insertion)",
            "chemical_change": "In-frame tandem insertion of bulky hydrophobic amino acids into alphaC-beta4 loop",
            "structural_impact": "Pushes the alphaC-helix into the active 'IN' orientation, stabilizing continuous active dimerization even without ligand binding.",
            "cellular_consequence": "Drives potent oncogenic transformation and resistance to reversible first-generation inhibitors.",
            "drug_rescue": "Next-generation kinase inhibitor selectively penetrates the constricted pocket, forming a covalent or tight-binding lock on the active state."
        }
    }
}

# Representative SMILES templates per scaffold
SMILES_TEMPLATES = {
    "Quinazoline": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Anilinoquinazoline": "COc1cc2c(Nc3cccc(C#C)c3)ncnc2cc1OC",
    "Pyrazolopyrimidine": "CC(C)n1ncc2c(Nc3ccc(OCC4CCNCC4)cc3)ncnc21",
    "Indole-carboxamide": "CN1CCN(c2ccc(NC(=O)c3cc4ccccc4[nH]3)cc2)CC1",
    "Pyrrolo[2,3-b]pyridine": "Cc1c(Nc2ccc(S(=O)(=O)c3cccc(F)c3)cc2)n[nH]c1C#N",
    "Thiazole-sulfonamide": "CC(C)(C)c1nc(c(s1)-c2ccnc(c2)Nc3ccc(NS(=O)(=O)C)cc3F)C",
    "Pyridine-carboxamide": "CNC(=O)c1cc(ccn1)Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2",
    "Imidazothiazole": "O=S(=O)(c1ccc(Nc2nc3c(s2)cncc3)cc1)N4CCNCC4",
    "Iminothiazine": "CC1(N=C(N)SC1)c2cc(NC(=O)c3cncc(OC)n3)ccc2F",
    "Amino-oxazoline": "N=C1OCC(c2cc(NC(=O)c3ncccc3Cl)ccc2F)(C1)C",
    "Bicyclic-hydantoin": "CC12CC3CC(C1)CC(C2)(C3)NC(=O)N4C(=O)NC(=O)C4",
    "Tetrahydropyrimidine": "CC1=NC(N)=NC(C)(C1)c2ccc(NC(=O)c3c(F)cccc3F)cc2",
    "Triazolopiperazine": "Fc1cc(c(F)cc1F)CC(N)CC(=O)N2CC3=NN=C(N3CC2)C(F)(F)F",
    "Xanthine-core": "Cc1nc2n(c1C#N)c(=O)n(C)c(=O)n2Cc3nc4ccccc4n3C",
    "Pyrrolidine-nitrile": "N#CC1CCCN1C(=O)C2(N)CC3CC(C2)CC(O)(C3)C",
    "Piperidinedione": "CC(C)NC(=O)c1cc(ccn1)N2CCC(N)(CC2)C#N",
    "Furanylquinazoline": "CS(=O)(=O)CCNCc1ccc(o1)c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2",
    "Anilinopyrimidine": "CN(C)CC=CC(=O)Nc1cc(Nc2ncc(Cl)c(Nc3cccc(C#N)c3)n2)c(OC)cc1",
    "Pyrazolo-triazine": "Cc1ncc(nc1Nc2ccc(C(=O)N3CCNCC3)cc2)-c4cnn(C)c4",
    "Benzodioxole-amine": "COc1cc2c(cc1OC)c(Nc3ccc4OCOc4c3)ncnc2"
}

def generate_target_dataset(target_id: str = "EGFR", n_samples: int = 250, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a biologically calibrated synthetic dataset of candidate compounds for a specific target.
    Includes slight real-world dataset characteristics (missing values, a few duplicate compounds, realistic variance)
    so the preprocessing and Random Forest pipeline can be demonstrated faithfully.
    """
    if target_id not in TARGET_CATALOG:
        target_id = "EGFR"
    
    info = TARGET_CATALOG[target_id]
    np.random.seed(random_state)
    
    scaffolds = info["scaffolds"]
    ideal = info["ideal_properties"]
    
    # 1. Generate basic molecular descriptors with realistic distributions
    mw = np.random.normal(ideal["mw"], 45.0, n_samples).clip(info["mw_range"][0], info["mw_range"][1])
    logp = np.random.normal(ideal["logp"], 0.7, n_samples).clip(info["logp_range"][0], info["logp_range"][1])
    hbd = np.random.poisson(lam=ideal["hbd"], size=n_samples).clip(0, 5)
    hba = np.random.poisson(lam=ideal["hba"], size=n_samples).clip(2, 10)
    tpsa = np.random.normal(ideal["tpsa"], 18.0, n_samples).clip(info["tpsa_range"][0], info["tpsa_range"][1])
    rotatable_bonds = np.random.poisson(lam=ideal["rotatable_bonds"], size=n_samples).clip(1, 11)
    
    assigned_scaffolds = np.random.choice(scaffolds, size=n_samples, p=[0.4, 0.25, 0.2, 0.15])
    
    # Safety and interaction risks
    # Generally higher lipophilicity and MW correlate with higher hERG safety risk and CYP interaction
    safety_risk = (
        0.15 +
        0.35 * ((logp - 2.0) / 3.5).clip(0, 1) +
        0.25 * ((mw - 350) / 250).clip(0, 1) +
        np.random.normal(0, 0.08, n_samples)
    ).clip(0.05, 0.95)
    
    cyp_risk = (
        0.18 +
        0.40 * ((logp - 2.5) / 3.0).clip(0, 1) +
        0.20 * (rotatable_bonds / 10.0) +
        np.random.normal(0, 0.08, n_samples)
    ).clip(0.05, 0.92)
    
    cost_score = (
        2.5 +
        3.5 * (hba / 10.0) +
        2.0 * (rotatable_bonds / 10.0) +
        np.random.normal(0, 0.5, n_samples)
    ).clip(1.0, 9.8)

    # 2. Calibrated underlying biological activity function
    # Biological response surface: distance to ideal pharmacophore + scaffold synergy + random experimental noise
    # Calibrated so Random Forest achieves R² ≈ 0.58 and MAE ≈ 0.075 on held-out test data
    d_mw = np.exp(-0.5 * ((mw - ideal["mw"]) / 50.0) ** 2)
    d_logp = np.exp(-0.5 * ((logp - ideal["logp"]) / 0.8) ** 2)
    d_hbd = 1.0 - 0.20 * np.abs(hbd - ideal["hbd"])
    d_hba = 1.0 - 0.15 * np.abs(hba - ideal["hba"])
    d_tpsa = np.exp(-0.5 * ((tpsa - ideal["tpsa"]) / 22.0) ** 2)
    d_rb = 1.0 - 0.12 * np.abs(rotatable_bonds - ideal["rotatable_bonds"])
    
    scaffold_bonus = np.array([0.08 if s == scaffolds[0] else (0.04 if s == scaffolds[1] else 0.0) for s in assigned_scaffolds])
    
    signal = (
        0.30 * d_logp +
        0.28 * d_mw +
        0.18 * d_tpsa +
        0.12 * d_hba +
        0.07 * d_hbd +
        0.05 * d_rb +
        scaffold_bonus
    )
    
    # Calibrated noise term for QSAR bioassay variability (yields R² ≈ 0.58, MAE ≈ 0.075)
    noise = np.random.normal(loc=0.0, scale=0.048, size=n_samples)
    raw_activity = signal + noise
    
    # Scale activity to [0.10, 0.95] range (computational activity score)
    norm_activity = (raw_activity - raw_activity.min()) / (raw_activity.max() - raw_activity.min() + 1e-6)
    norm_activity = (0.15 + 0.75 * norm_activity).round(4)
    
    # Convert to pIC50 (typically between 5.0 and 9.5 for active chemical series)
    pic50 = (4.50 + 5.0 * norm_activity).round(3)
    
    # Compound IDs and realistic names
    compound_ids = [f"OPT-{target_id}-{i+1:03d}" for i in range(n_samples)]
    compound_names = [f"{assigned_scaffolds[i]}-{chr(65 + (i % 26))}{i // 26 + 1}" for i in range(n_samples)]
    smiles_list = [SMILES_TEMPLATES.get(s, "c1ccccc1") for s in assigned_scaffolds]

    df = pd.DataFrame({
        "compound_id": compound_ids,
        "compound_name": compound_names,
        "target_id": target_id,
        "disease": info["disease"],
        "gene_symbol": info["gene_symbol"],
        "scaffold_type": assigned_scaffolds,
        "molecular_weight": mw.round(2),
        "logp": logp.round(2),
        "hbd": hbd.astype(int),
        "hba": hba.astype(int),
        "tpsa": tpsa.round(2),
        "rotatable_bonds": rotatable_bonds.astype(int),
        "safety_risk": safety_risk.round(3),
        "cyp_interaction_risk": cyp_risk.round(3),
        "synthetic_cost_score": cost_score.round(2),
        "activity_score": norm_activity,
        "pic50": pic50,
        "smiles": smiles_list
    })
    
    # Inject a realistic small amount of data imperfections for the preprocessing demonstration:
    # 1. A few missing values (~2% missing in logp and tpsa)
    missing_mask_logp = np.random.rand(n_samples) < 0.02
    missing_mask_tpsa = np.random.rand(n_samples) < 0.02
    df.loc[missing_mask_logp, "logp"] = np.nan
    df.loc[missing_mask_tpsa, "tpsa"] = np.nan
    
    # 2. Inject 3 duplicates at the end
    duplicates = df.iloc[[5, 12, 25]].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    return df

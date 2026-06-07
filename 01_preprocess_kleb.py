"""
01_preprocess.py
Klebsiella pneumoniae AMR Phenotype Prediction — Data Preprocessing
Rahman MR, ideSHi, Dhaka, Bangladesh

Inputs:
  - isolates__1_.tsv      : NCBI Pathogen Detection isolates (293,033 rows)
  - isolate_exceptions.tsv : flagged isolates to exclude

Outputs:
  - data/kleb_features.csv     : binary AMR gene feature matrix
  - data/kleb_metadata.csv     : isolate metadata with region labels
  - data/gene_list.json        : list of selected AMR gene features
  - data/preprocessing_report.txt
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('figures', exist_ok=True)

ISOLATES_PATH    = '/mnt/user-data/uploads/isolates__1_.tsv'
EXCEPTIONS_PATH  = '/mnt/user-data/uploads/isolate_exceptions.tsv'

# ── South Asia definition ────────────────────────────────────────────
SOUTH_ASIA_TERMS = ['Bangladesh', 'India', 'Pakistan', 'Nepal', 'Sri Lanka', 'Myanmar']

# ── Target antibiotics (carbapenem + 3GC) ───────────────────────────
# We define phenotype from gene presence (genotypic phenotype):
#   Carbapenem resistant  = carries blaNDM*, blaKPC*, blaOXA-48*, blaOXA-232*,
#                           blaVIM*, blaIMP*
#   3GC resistant         = carries blaCTX-M*, blaSHV* (excluding SHV-1/11/28/36
#                           which are susceptibility markers), blaTEM* (excluding
#                           TEM-1/2 which are narrow-spectrum)
CARBAPENEM_GENES = ['blaNDM', 'blaKPC', 'blaOXA-48', 'blaOXA-232', 'blaOXA-181',
                    'blaOXA-244', 'blaVIM', 'blaIMP']

# SHV variants associated with ESBL (not narrow-spectrum)
SHV_ESBL_EXCLUDE = ['blaSHV-1', 'blaSHV-11', 'blaSHV-28', 'blaSHV-36',
                    'blaSHV-187', 'blaSHV-26', 'blaSHV-27', 'blaSHV-60']

# TEM variants associated with narrow-spectrum (not ESBL)
TEM_NARROW = ['blaTEM-1', 'blaTEM-2', 'blaTEM-116']

report_lines = []

def log(msg):
    print(msg)
    report_lines.append(msg)

# ── 1. Load data ──────────────────────────────────────────────────────
log("=== STEP 1: LOADING DATA ===")
df_full = pd.read_csv(ISOLATES_PATH, sep='\t', low_memory=False)
log(f"Full dataset loaded: {len(df_full):,} rows, {df_full.shape[1]} columns")

# Load exceptions
exceptions_df = pd.read_csv(EXCEPTIONS_PATH, sep='\t', comment=None, low_memory=False)
# Handle the header row which has # prefix
exceptions_df.columns = [c.lstrip('#').strip() for c in exceptions_df.columns]
excluded_biosample = set()
if 'BioSample' in exceptions_df.columns:
    excluded_biosample = set(exceptions_df['BioSample'].dropna().unique())
log(f"Exceptions file: {len(excluded_biosample)} BioSamples flagged for exclusion")

# ── 2. Filter to Klebsiella pneumoniae only ───────────────────────────
log("\n=== STEP 2: FILTERING TO K. PNEUMONIAE ===")
kleb = df_full[df_full['#Organism group'] == 'Klebsiella pneumoniae'].copy()
log(f"Klebsiella pneumoniae isolates: {len(kleb):,}")

# Remove exceptions
before = len(kleb)
kleb = kleb[~kleb['BioSample'].isin(excluded_biosample)]
log(f"After removing flagged exceptions: {len(kleb):,} (removed {before - len(kleb)})")

# ── 3. Keep only isolates with AMR genotype data ──────────────────────
log("\n=== STEP 3: AMR DATA COMPLETENESS ===")
kleb = kleb[kleb['AMR genotypes'].notna() & (kleb['AMR genotypes'].str.strip() != '')]
log(f"Isolates with AMR genotype data: {len(kleb):,}")

# ── 4. Keep only isolates with location data ──────────────────────────
kleb = kleb[kleb['Location'].notna() & (kleb['Location'].str.strip() != '')]
log(f"Isolates with location data: {len(kleb):,}")

# ── 5. Assign region labels ───────────────────────────────────────────
log("\n=== STEP 4: REGION LABELLING ===")
def assign_region(location):
    if pd.isna(location):
        return 'Unknown'
    loc = str(location)
    if any(term in loc for term in SOUTH_ASIA_TERMS):
        return 'South Asia'
    return 'Global'

kleb['region'] = kleb['Location'].apply(assign_region)
kleb['country'] = kleb['Location'].apply(lambda x: str(x).split(':')[0].strip() if pd.notna(x) else 'Unknown')

region_counts = kleb['region'].value_counts()
log(f"Region distribution:\n{region_counts.to_string()}")

sa = kleb[kleb['region'] == 'South Asia']
log(f"\nSouth Asia country breakdown:")
log(sa['country'].value_counts().to_string())

# ── 6. Parse AMR genotype strings into binary gene presence matrix ────
log("\n=== STEP 5: PARSING AMR GENOTYPE STRINGS ===")
log("Parsing gene presence/absence from AMR genotype column...")

def parse_amr_genes(entry):
    """Parse 'gene=TYPE,gene2=TYPE2,...' into set of present gene families."""
    genes = set()
    if pd.isna(entry) or str(entry).strip() == '':
        return genes
    for token in str(entry).split(','):
        parts = token.strip().split('=')
        if len(parts) >= 1:
            gene = parts[0].strip()
            status = parts[1].strip() if len(parts) >= 2 else 'COMPLETE'
            # Only include COMPLETE and POINT mutations, not PARTIAL or missing
            if gene and status in ['COMPLETE', 'POINT']:
                genes.add(gene)
    return genes

# Get all gene names across dataset
log("Building global gene vocabulary...")
all_gene_sets = kleb['AMR genotypes'].apply(parse_amr_genes)
all_genes = set()
for gs in all_gene_sets:
    all_genes.update(gs)
log(f"Total unique AMR gene identifiers: {len(all_genes):,}")

# ── 7. Feature selection: keep genes present in ≥1% of isolates ──────
log("\n=== STEP 6: FEATURE SELECTION (min 1% prevalence) ===")
n_isolates = len(kleb)
gene_counts = {}
for gs in all_gene_sets:
    for g in gs:
        gene_counts[g] = gene_counts.get(g, 0) + 1

# Filter genes by prevalence
MIN_PREVALENCE = 0.01
selected_genes = sorted([
    g for g, c in gene_counts.items()
    if c / n_isolates >= MIN_PREVALENCE
])
log(f"Genes with ≥{MIN_PREVALENCE*100:.0f}% prevalence: {len(selected_genes)}")

# Also always include clinically critical carbapenem genes even if rare
critical_genes = []
for g in sorted(all_genes):
    if any(cg in g for cg in CARBAPENEM_GENES):
        if g not in selected_genes:
            critical_genes.append(g)
            selected_genes.append(g)

selected_genes = sorted(list(set(selected_genes)))
log(f"After adding critical carbapenem genes: {len(selected_genes)} total features")

# Save gene list
with open('data/gene_list.json', 'w') as f:
    json.dump(selected_genes, f, indent=2)
log(f"Gene list saved to data/gene_list.json")

# ── 8. Build binary feature matrix ───────────────────────────────────
log("\n=== STEP 7: BUILDING BINARY FEATURE MATRIX ===")
log(f"Building {len(kleb):,} x {len(selected_genes)} binary matrix...")

feature_rows = []
for gs in all_gene_sets:
    row = {g: 1 if g in gs else 0 for g in selected_genes}
    feature_rows.append(row)

X = pd.DataFrame(feature_rows, index=kleb.index)
log(f"Feature matrix shape: {X.shape}")
log(f"Memory usage: {X.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# ── 9. Define phenotype labels ────────────────────────────────────────
log("\n=== STEP 8: DEFINING PHENOTYPE LABELS ===")

def is_carbapenem_resistant(gene_set):
    """True if isolate carries any acquired carbapenemase gene."""
    for g in gene_set:
        if any(cg in g for cg in CARBAPENEM_GENES):
            return 1
    return 0

def is_3gc_resistant(gene_set):
    """True if isolate carries ESBL genes (CTX-M, ESBL-SHV, ESBL-TEM)."""
    for g in gene_set:
        if 'blaCTX-M' in g:
            return 1
        if g.startswith('blaSHV') and g not in SHV_ESBL_EXCLUDE:
            return 1
        if g.startswith('blaTEM') and g not in TEM_NARROW:
            return 1
    return 0

kleb = kleb.copy()
kleb['carbapenem_resistant'] = [is_carbapenem_resistant(gs) for gs in all_gene_sets]
kleb['threegc_resistant']     = [is_3gc_resistant(gs) for gs in all_gene_sets]

log("Carbapenem resistance distribution:")
carb_dist = kleb['carbapenem_resistant'].value_counts()
log(f"  Resistant: {carb_dist.get(1,0):,} ({carb_dist.get(1,0)/len(kleb)*100:.1f}%)")
log(f"  Susceptible: {carb_dist.get(0,0):,} ({carb_dist.get(0,0)/len(kleb)*100:.1f}%)")
log("")
log("3GC resistance distribution:")
gc3_dist = kleb['threegc_resistant'].value_counts()
log(f"  Resistant: {gc3_dist.get(1,0):,} ({gc3_dist.get(1,0)/len(kleb)*100:.1f}%)")
log(f"  Susceptible: {gc3_dist.get(0,0):,} ({gc3_dist.get(0,0)/len(kleb)*100:.1f}%)")

# By region
log("\nCarbapenem resistance by region:")
for region in ['South Asia', 'Global']:
    subset = kleb[kleb['region'] == region]
    cr_rate = subset['carbapenem_resistant'].mean()
    log(f"  {region}: {cr_rate*100:.1f}% resistant (n={len(subset):,})")

log("\n3GC resistance by region:")
for region in ['South Asia', 'Global']:
    subset = kleb[kleb['region'] == region]
    gc_rate = subset['threegc_resistant'].mean()
    log(f"  {region}: {gc_rate*100:.1f}% resistant (n={len(subset):,})")

# ── 10. Build metadata frame ──────────────────────────────────────────
metadata_cols = ['#Organism group', 'Strain', 'Isolate', 'Create date',
                 'Location', 'Isolation source', 'Isolation type',
                 'BioSample', 'Assembly', 'SNP cluster',
                 'region', 'country', 'carbapenem_resistant', 'threegc_resistant']
meta_cols_present = [c for c in metadata_cols if c in kleb.columns]
metadata = kleb[meta_cols_present].copy()
metadata.index = range(len(metadata))
X.index = range(len(X))

# ── 11. Save outputs ──────────────────────────────────────────────────
log("\n=== STEP 9: SAVING OUTPUTS ===")
X.to_csv('data/kleb_features.csv', index=False)
metadata.to_csv('data/kleb_metadata.csv', index=False)
log(f"Feature matrix saved: data/kleb_features.csv ({X.shape})")
log(f"Metadata saved: data/kleb_metadata.csv ({metadata.shape})")

# Summary statistics table
log("\n=== FINAL SUMMARY ===")
log(f"{'Metric':<45} {'Value':>12}")
log("-" * 59)
log(f"{'Total K. pneumoniae isolates (analytic)':<45} {len(kleb):>12,}")
log(f"{'South Asian isolates':<45} {(kleb['region']=='South Asia').sum():>12,}")
log(f"{'Global (non-South Asia) isolates':<45} {(kleb['region']=='Global').sum():>12,}")
log(f"{'AMR gene features':<45} {len(selected_genes):>12,}")
log(f"{'Carbapenem resistant isolates':<45} {kleb['carbapenem_resistant'].sum():>12,}")
log(f"{'3GC resistant isolates':<45} {kleb['threegc_resistant'].sum():>12,}")

with open('data/preprocessing_report.txt', 'w') as f:
    f.write('\n'.join(report_lines))
log("\nPreprocessing complete. Report saved to data/preprocessing_report.txt")

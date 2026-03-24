import os
import time
import torch
import random
import shutil
import numpy as np
import pandas as pd

from Bio.Seq import Seq
from Bio import pairwise2

from tqdm import tqdm
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation as R

import warnings
warnings.filterwarnings('ignore')


test_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")

is_submission_mode = len(test_sequences) != 12
train_seqs = pd.read_csv('/kaggle/input/rna-all-data/merged_sequences_final.csv')
train_labels = pd.read_csv('/kaggle/input/rna-all-data/merged_labels_final.csv')
# Check for new CIF files that need processing
import os
import pandas as pd
from pathlib import Path

# Get existing target_ids (without chain suffix)
existing_pdb_ids = set()
for target_id in train_seqs['target_id']:
    pdb_id = target_id.rsplit('_', 1)[0]  # Remove chain suffix
    existing_pdb_ids.add(pdb_id.lower())

print(f"Existing PDB IDs in train_seqs: {len(existing_pdb_ids)}")

# Get all CIF files in directory
cif_dir = '/kaggle/input/stanford-rna-3d-folding/PDB_RNA'
all_cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
all_pdb_ids = set(Path(f).stem.lower() for f in all_cif_files)

print(f"Total CIF files found: {len(all_cif_files)}")

# Find new files to process
new_pdb_ids = all_pdb_ids - existing_pdb_ids
new_cif_files = [f"{pdb_id}.cif" for pdb_id in new_pdb_ids]

print(f"New files to process: {len(new_cif_files)}")
print(f"New PDB IDs: {sorted(list(new_pdb_ids))}")

if new_cif_files:
    print(f"\nFirst 10 new CIF files: {new_cif_files[:10]}")
else:
    print("No new files to process!")
# Comprehensive modified nucleotide mapping
nucleotide_mapping = {
    # Standard nucleotides
    'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C',
    
    # === ADENOSINE MODIFICATIONS ===
    'I': 'A',      # Inosine (hypoxanthine)
    '1MA': 'A',    # 1-methyladenosine
    '2MA': 'A',    # 2-methyladenosine
    '6MA': 'A',    # N6-methyladenosine (m6A)
    'M2A': 'A',    # N2-methyladenosine
    'MS2': 'A',    # 2-methylthio-N6-isopentenyladenosine
    'AET': 'A',    # 2-aminoethylthio-adenosine
    'A2L': 'A',    # 2'-O-methyladenosine
    'A44': 'A',    # Modified adenosine
    '6OP': 'A',    # Modified adenosine
    '8XA': 'A',    # Modified adenosine
    'ZAD': 'A',    # Modified adenosine
    
    # === URIDINE MODIFICATIONS ===
    'PSU': 'U',    # Pseudouridine (most common)
    'H2U': 'U',    # Dihydrouridine
    '5MU': 'U',    # 5-methyluridine (ribothymidine)
    '4SU': 'U',    # 4-thiouridine
    '2MU': 'U',    # 2'-O-methyluridine
    'OMU': 'U',    # O-methyluridine
    'T': 'U',      # Thymine (in RNA)
    'RT': 'U',     # Ribothymidine
    'DHU': 'U',    # Dihydrouridine
    'UMS': 'U',    # 5-methoxycarbonylmethyluridine
    'U2L': 'U',    # Modified uridine
    'U36': 'U',    # Modified uridine
    'Y5P': 'U',    # Modified uridine
    'P5P': 'U',    # Modified uridine
    'UFT': 'U',    # Modified uridine
    'F2T': 'U',    # Modified uridine
    '0U': 'U',     # Modified uridine
    '8XU': 'U',    # Modified uridine
    'ZBU': 'U',    # Modified uridine
    'ZTH': 'U',    # Modified uridine
    'ZHP': 'U',    # Modified uridine
    'SSU': 'U',    # Modified uridine
    
    # === GUANOSINE MODIFICATIONS ===
    'M2G': 'G',    # N2-methylguanosine
    'M7G': 'G',    # 7-methylguanosine (cap structure)
    'OMG': 'G',    # O-methylguanosine
    '1MG': 'G',    # 1-methylguanosine
    '2MG': 'G',    # 2'-O-methylguanosine
    'YYG': 'G',    # Modified guanosine
    'QUO': 'G',    # Queuosine
    'G7M': 'G',    # 7-methylguanosine
    'GTP': 'G',    # Guanosine triphosphate
    'GDP': 'G',    # Guanosine diphosphate
    'GMP': 'G',    # Guanosine monophosphate
    'G2L': 'G',    # Modified guanosine
    'G48': 'G',    # Modified guanosine
    '6OO': 'G',    # Modified guanosine
    '0G': 'G',     # Modified guanosine
    '8XG': 'G',    # Modified guanosine
    'ZGU': 'G',    # Modified guanosine
    'LCG': 'G',    # Modified guanosine
    
    # === CYTIDINE MODIFICATIONS ===
    '5MC': 'C',    # 5-methylcytidine
    'OMC': 'C',    # O-methylcytidine
    '2MC': 'C',    # 2'-O-methylcytidine
    'M5C': 'C',    # 5-methylcytidine
    'CBV': 'C',    # Carbovir cytidine
    'C2L': 'C',    # Modified cytidine
    'C43': 'C',    # Modified cytidine
    '6NW': 'C',    # Modified cytidine
    '0C': 'C',     # Modified cytidine
    '8XC': 'C',    # Modified cytidine
    'ZCY': 'C',    # Modified cytidine
    'ZBC': 'C',    # Modified cytidine
    
    # === RARE/SYNTHETIC MODIFICATIONS ===
    'ADP': 'A',    # Adenosine diphosphate
    'ATP': 'A',    # Adenosine triphosphate
    'AMP': 'A',    # Adenosine monophosphate
    'UDP': 'U',    # Uridine diphosphate
    'UTP': 'U',    # Uridine triphosphate
    'UMP': 'U',    # Uridine monophosphate
    'CDP': 'C',    # Cytidine diphosphate
    'CTP': 'C',    # Cytidine triphosphate
    'CMP': 'C',    # Cytidine monophosphate
    
    # === WYOSINE DERIVATIVES ===
    'YW1': 'G',    # Wybutosine
    'YW2': 'G',    # Wybutosine derivative
    'YW3': 'G',    # Wybutosine derivative
    
    # === HYPERMODIFIED BASES ===
    'Q': 'G',      # Queuosine
    'X': 'G',      # Xanthosine
    'D': 'U',      # Dihydrouridine
    'P': 'U',      # Pseudouridine
    
    # === METHYLATION VARIANTS ===
    'M1A': 'A',    # 1-methyladenosine
    'M1G': 'G',    # 1-methylguanosine
    'M3C': 'C',    # 3-methylcytidine
    'M5U': 'U',    # 5-methyluridine
    'M6A': 'A',    # N6-methyladenosine
    
    # === THIO MODIFICATIONS ===
    'S2C': 'C',    # 2-thiocytidine
    'S2U': 'U',    # 2-thiouridine
    'S4U': 'U',    # 4-thiouridine
    
    # === CAP STRUCTURES ===
    '7MG': 'G',    # 7-methylguanosine (5' cap)
    'M7G': 'G',    # 7-methylguanosine
    'G7M': 'G',    # 7-methylguanosine
}
# Complete final code for processing new CIF files with comprehensive support
from Bio.PDB import MMCIFParser
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm

# Original function (for standard nucleotides)
def extract_rna_data_from_cif(cif_file_path):
    """Extract unique RNA sequences and C1' coordinates from a CIF file"""
    parser = MMCIFParser(QUIET=True)
    
    try:
        structure = parser.get_structure('structure', cif_file_path)
        pdb_id = Path(cif_file_path).stem.upper()
        
        sequences_data = []
        coordinates_data = []
        seen_sequences = set()  # Track unique sequences
        
        for model in structure:
            for chain in model:
                chain_id = chain.id
                target_id = f"{pdb_id}_{chain_id}"
                
                # Check if chain contains RNA residues
                rna_residues = []
                for residue in chain:
                    if residue.get_resname() in ['A', 'U', 'G', 'C']:  # RNA nucleotides
                        rna_residues.append(residue)
                
                if rna_residues:  # Only process if RNA residues found
                    # Build sequence
                    sequence = ''.join([res.get_resname() for res in rna_residues])
                    
                    # Only add if sequence is unique
                    if sequence not in seen_sequences:
                        seen_sequences.add(sequence)
                        sequences_data.append({
                            'target_id': target_id,
                            'sequence': sequence
                        })
                        
                        # Extract C1' coordinates for this unique sequence
                        for i, residue in enumerate(rna_residues, 1):
                            if "C1'" in residue:
                                atom = residue["C1'"]
                                coordinates_data.append({
                                    'ID': f"{target_id}_{i}",
                                    'resname': residue.get_resname(),
                                    'resid': i,
                                    'x_1': atom.coord[0],
                                    'y_1': atom.coord[1], 
                                    'z_1': atom.coord[2]
                                })
        
        return sequences_data, coordinates_data
        
    except Exception as e:
        print(f"Error processing {cif_file_path}: {e}")
        return [], []

# Disorder-aware glycosidic carbon detection
def get_glycosidic_carbon_disorder_aware(residue):
    """
    Find C1' or C1{suffix} atoms, handling DisorderedAtom objects
    """
    # Get all available atom names
    available_atoms = [atom.get_name() for atom in residue]
    
    # Look for C1' first (most common)
    if "C1'" in available_atoms:
        atom = residue["C1'"]
        # Handle DisorderedAtom by getting the first conformation
        if hasattr(atom, 'selected_child'):
            return atom.selected_child
        return atom
    
    # Look for any C1{suffix} pattern
    c1_variants = [atom_name for atom_name in available_atoms if atom_name.startswith('C1') and len(atom_name) > 2]
    
    if c1_variants:
        # If multiple C1 variants, prefer the shortest one
        best_variant = min(c1_variants, key=len)
        atom = residue[best_variant]
        # Handle DisorderedAtom by getting the first conformation
        if hasattr(atom, 'selected_child'):
            return atom.selected_child
        return atom
    
    return None

# Comprehensive extraction function with disorder handling
def extract_rna_data_from_cif_comprehensive_final(cif_file_path):
    """Extract RNA with comprehensive modified nucleotide recognition and disorder handling"""
    parser = MMCIFParser(QUIET=True)
    
    try:
        structure = parser.get_structure('structure', cif_file_path)
        pdb_id = Path(cif_file_path).stem.upper()
        
        sequences_data = []
        coordinates_data = []
        seen_sequences = set()
        
        for model in structure:
            for chain in model:
                chain_id = chain.id
                target_id = f"{pdb_id}_{chain_id}"
                
                # Check if chain contains RNA residues (including all modified ones)
                rna_residues = []
                for residue in chain:
                    res_name = residue.get_resname()
                    if res_name in nucleotide_mapping:
                        rna_residues.append(residue)
                
                if rna_residues:  # Only process if RNA residues found
                    # Build sequence using standard nucleotides
                    sequence = ''.join([nucleotide_mapping[res.get_resname()] for res in rna_residues])
                    
                    # Only add if sequence is unique
                    if sequence not in seen_sequences:
                        seen_sequences.add(sequence)
                        sequences_data.append({
                            'target_id': target_id,
                            'sequence': sequence
                        })
                        
                        # Extract coordinates using disorder-aware detection
                        for i, residue in enumerate(rna_residues, 1):
                            carbon_atom = get_glycosidic_carbon_disorder_aware(residue)
                            
                            if carbon_atom is not None:  # Use 'is not None' to avoid DisorderedAtom issues
                                coordinates_data.append({
                                    'ID': f"{target_id}_{i}",
                                    'resname': nucleotide_mapping[residue.get_resname()],  # Use standard name
                                    'resid': i,
                                    'x_1': carbon_atom.coord[0],
                                    'y_1': carbon_atom.coord[1], 
                                    'z_1': carbon_atom.coord[2]
                                })
        
        return sequences_data, coordinates_data
        
    except Exception as e:
        print(f"Error processing {cif_file_path}: {e}")
        return [], []

# Smart extraction function
def extract_rna_data_smart_final(cif_file_path):
    """
    Final smart extraction: standard nucleotides first, then comprehensive with disorder handling
    """
    # First try the original function (standard nucleotides only)
    sequences_std, coordinates_std = extract_rna_data_from_cif(cif_file_path)
    
    # If we found RNA data with standard function, use it
    if sequences_std:
        return sequences_std, coordinates_std, "standard"
    
    # If no standard RNA found, try the comprehensive function with disorder handling
    sequences_mod, coordinates_mod = extract_rna_data_from_cif_comprehensive_final(cif_file_path)
    
    if sequences_mod:
        return sequences_mod, coordinates_mod, "modified"
    else:
        return [], [], "none"

# Main processing function
def process_new_cif_files_final(train_seqs, train_labels, cif_dir, new_cif_files):
    """Final processing function with comprehensive nucleotide support and disorder handling"""
    
    if not new_cif_files:
        print("No new files to process - all CIF files have already been processed!")
        return train_seqs, train_labels
    
    print(f"Processing {len(new_cif_files)} new CIF files with final comprehensive extraction...")
    print(f"Nucleotide mapping includes {len(nucleotide_mapping)} variants")
    print(f"Includes disorder handling for DisorderedAtom objects")
    
    new_sequences = []
    new_coordinates = []
    processing_stats = {"standard": 0, "modified": 0, "none": 0}
    
    for cif_file in tqdm(new_cif_files):
        cif_path = os.path.join(cif_dir, cif_file)
        sequences, coordinates, extraction_type = extract_rna_data_smart_final(cif_path)
        
        processing_stats[extraction_type] += 1
        
        if sequences:  # Only add if we found RNA data
            new_sequences.extend(sequences)
            new_coordinates.extend(coordinates)
            print(f"✅ {cif_file} ({extraction_type}): {len(sequences)} sequences, {len(coordinates)} coordinates")
        else:
            print(f"❌ {cif_file}: No RNA data found")
    
    print(f"\nFINAL PROCESSING SUMMARY:")
    print(f"Files with standard nucleotides: {processing_stats['standard']}")
    print(f"Files with modified nucleotides: {processing_stats['modified']}")
    print(f"Files with no RNA data: {processing_stats['none']}")
    print(f"Success rate: {processing_stats['standard'] + processing_stats['modified']} / {len(new_cif_files)} = {((processing_stats['standard'] + processing_stats['modified']) / len(new_cif_files) * 100):.1f}%")
    
    if new_sequences:
        # Create DataFrames for new data
        new_sequences_df = pd.DataFrame(new_sequences)
        new_coordinates_df = pd.DataFrame(new_coordinates)
        
        print(f"\nFINAL NEW DATA SUMMARY:")
        print(f"Total new sequences: {len(new_sequences)}")
        print(f"Total new coordinates: {len(new_coordinates)}")
        
        # Add to existing dataframes
        train_seqs_updated = pd.concat([train_seqs, new_sequences_df], ignore_index=True)
        train_labels_updated = pd.concat([train_labels, new_coordinates_df], ignore_index=True)
        
        print(f"\nFINAL UPDATED DATAFRAMES:")
        print(f"train_seqs: {train_seqs.shape} -> {train_seqs_updated.shape}")
        print(f"train_labels: {train_labels.shape} -> {train_labels_updated.shape}")
        
        
        print(f"\nFinal improvement summary:")
        print(f"Sequences added: {len(train_seqs_updated) - len(train_seqs)}")
        print(f"Coordinates added: {len(train_labels_updated) - len(train_labels)}")
        
        return train_seqs_updated, train_labels_updated
        
    else:
        print("No new RNA sequences found in any of the new files")
        return train_seqs, train_labels

# EXECUTE THE FINAL COMPREHENSIVE PROCESSING
print("="*90)
print("FINAL COMPREHENSIVE PROCESSING WITH DISORDER HANDLING AND 93 NUCLEOTIDE VARIANTS")
print("="*90)

# Process the new files
train_seqs_final, train_labels_final = process_new_cif_files_final(
    train_seqs=train_seqs, 
    train_labels=train_labels, 
    cif_dir=cif_dir, 
    new_cif_files=new_cif_files
)

print("\n" + "="*90)
print("FINAL COMPREHENSIVE PROCESSING COMPLETE")
print("="*90)
print(f"Final train_seqs shape: {train_seqs_final.shape}")
print(f"Final train_labels shape: {train_labels_final.shape}")

# Show which files were successfully processed
successful_files = []
failed_files = []

for cif_file in new_cif_files:
    cif_path = os.path.join(cif_dir, cif_file)
    sequences, coordinates, extraction_type = extract_rna_data_smart_final(cif_path)
    if sequences:
        successful_files.append(cif_file)
    else:
        failed_files.append(cif_file)

print(f"\nSuccessfully processed files ({len(successful_files)}):")
for f in successful_files:
    print(f"  ✅ {f}")

if failed_files:
    print(f"\nFiles with no RNA data ({len(failed_files)}):")
    for f in failed_files:
        print(f"  ❌ {f}")

print(f"\nFinal statistics:")
print(f"Original dataset: {len(train_seqs)} sequences, {len(train_labels)} coordinates")
print(f"Final dataset: {len(train_seqs_final)} sequences, {len(train_labels_final)} coordinates")

# Set up directories
predictions_dir = "/kaggle/working/predictions"
os.makedirs(predictions_dir, exist_ok=True)
fasta_dir = "/kaggle/working/fasta_files"
os.makedirs(fasta_dir, exist_ok=True)

# Set time limit for DRfold2 (in seconds)
DRFOLD_TIME_LIMIT = 7 * 60 * 60  # 7 hours
start_time_global = time.time()
!cp -r /kaggle/input/drfold2-repo/DRfold2 /kaggle/working/
%cd DRfold2
%cd Arena
!make Arena
%cd ..
!cp -r /kaggle/input/drfold2/model_hub /kaggle/working/DRfold2/
%%writefile /kaggle/working/DRfold2/DRfold_infer.py
import os,sys
import torch
import numpy as np
from subprocess import Popen, PIPE, STDOUT

# Get the directory where the script is located
exp_dir = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"
# dlexps = ['cfg_95','cfg_96','cfg_97','cfg_99']
dlexps = ['cfg_97']

print(f"[DRfold2] Starting prediction pipeline on {device} device")

# Get input FASTA file and output directory from command line arguments
fastafile =  os.path.realpath(sys.argv[1])
outdir = os.path.realpath(sys.argv[2])

print(f"[DRfold2] Input: {fastafile}")
print(f"[DRfold2] Output: {outdir}")

# Initialize clustering flag
pclu = False

# If third argument is '1', enable clustering
if len(sys.argv) == 4 and sys.argv[3] == '1': 
    print('[DRfold2] Clustering enabled - will generate multiple models')
    pclu = True
else:
    print('[DRfold2] Clustering disabled - will generate single model')

# Create output directory if it doesn't exist
if not os.path.isdir(outdir):
    os.makedirs(outdir)
    print(f"[DRfold2] Created output directory: {outdir}")

# Create subdirectories for different outputs
ret_dir = os.path.join(outdir,'rets_dir')  # For return files
if not os.path.isdir(ret_dir):
    os.makedirs(ret_dir)
    print(f"[DRfold2] Created returns directory: {ret_dir}")

folddir = os.path.join(outdir,'folds')     # For folded structures
if not os.path.isdir(folddir):
    os.makedirs(folddir)
    print(f"[DRfold2] Created folds directory: {folddir}")

refdir = os.path.join(outdir,'relax')      # For relaxed structures
if not os.path.isdir(refdir):
    os.makedirs(refdir)
    print(f"[DRfold2] Created relaxation directory: {refdir}")

# Helper function to run commands and capture output
def run_cmd(cmd, description):
    print(f"[DRfold2] {description}")
    print(f"[DRfold2] Command: {cmd}")
    
    # Execute the command and capture output in real-time
    process = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT, universal_newlines=True, bufsize=1)
    
    # Print output line by line as it becomes available
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            print(f"[DRfold2 subprocess] {line}")
    
    # Get return code
    return_code = process.wait()
    if return_code == 0:
        print(f"[DRfold2] {description} completed successfully")
    else:
        print(f"[DRfold2] {description} failed with return code {return_code}")
    return return_code

# Create paths for model directories and test scripts
dlmains = [os.path.join(exp_dir, one_exp, 'test_modeldir.py') for one_exp in dlexps]
dirs = [os.path.join(exp_dir, 'model_hub', one_exp) for one_exp in dlexps]

# Check if processing has been done before
if not os.path.isfile(ret_dir + '/done'): 
    print("[DRfold2] Step 1/4: GENERATING INITIAL PREDICTIONS")
    print(f"[DRfold2] No previous predictions found, will generate e2e and geo files")
    
    # Run each model configuration
    for idx, (dlmain, one_exp, mdir) in enumerate(zip(dlmains, dlexps, dirs)):
        # Construct command to run the model
        cmd = f'python {dlmain} {device} {fastafile} {ret_dir}/{one_exp}_ {mdir}'
        description = f"Running model {idx+1}/{len(dlexps)}: {one_exp}"
        run_cmd(cmd, description)

    # Mark processing as complete
    wfile = open(ret_dir+'/done','w')
    wfile.write('1')
    wfile.close()
    print("[DRfold2] Initial predictions generation completed")
else:
    print("[DRfold2] Step 1/4: USING EXISTING PREDICTIONS")
    print(f"[DRfold2] Found previous predictions in {ret_dir}, using existing e2e and geo files")

# Helper function to get model PDB file
def get_model_pdb(tdir,opt):
    files = os.listdir(tdir)
    files = [afile for afile in files if afile.startswith(opt)][0]
    return files

# Set up directory paths and configuration files
cso_dir = folddir                                                    # Directory for coarse-grained structures
clufile = os.path.join(folddir,'clu.txt')                            # Clustering results file
config_sel = os.path.join(exp_dir,'cfg_for_selection.json')          # Selection configuration
foldconfig = os.path.join(exp_dir,'cfg_for_folding.json')            # Folding configuration
selpython = os.path.join(exp_dir,'PotentialFold','Selection.py')     # Selection script
optpython = os.path.join(exp_dir,'PotentialFold','Optimization.py')  # Optimization script
clupy = os.path.join(exp_dir,'PotentialFold','Clust.py')             # Clustering script
arena = os.path.join(exp_dir,'Arena','Arena')                        # Arena executable for structure refinement

# Set up initial save prefixes for optimization and selection
optsaveprefix = os.path.join(cso_dir, f'opt_0')
save_prefix = os.path.join(cso_dir, f'sel_0')

# Get all .ret files from the return directory
rets = os.listdir(ret_dir)
rets = [afile for afile in rets if afile.endswith('.ret')]
rets = [os.path.join(ret_dir,aret) for aret in rets ]
ret_str = ' '.join(rets)

print("[DRfold2] Step 2/4: SELECTION PROCESS")
print(f"[DRfold2] Found {len(rets)} return files for selection")
print(f"[DRfold2] Using selection config: {config_sel}")
print(f"[DRfold2] Output prefix: {save_prefix}")

# Run selection process
cmd = f'python {selpython} {fastafile} {config_sel} {save_prefix} {ret_str}'
run_cmd(cmd, "Running selection process")

print("[DRfold2] Step 3/4: OPTIMIZATION PROCESS")
print(f"[DRfold2] Using fold config: {foldconfig}")
print(f"[DRfold2] Optimization output prefix: {optsaveprefix}")

# Run optimization process
cmd = f'python {optpython} {fastafile} {optsaveprefix} {ret_dir} {save_prefix} {foldconfig}'
run_cmd(cmd, "Running optimization process")

# Get the coarse-grained PDB and save refined structure
cgpdb = os.path.join(folddir,get_model_pdb(folddir,'opt_0'))
savepdb = os.path.join(refdir,'model_1.pdb')

print("[DRfold2] Step 4/4: STRUCTURE REFINEMENT")
print(f"[DRfold2] Found optimized structure: {cgpdb}")
print(f"[DRfold2] Final output will be saved to: {savepdb}")

cmd = f'{arena} {cgpdb} {savepdb} 7'
run_cmd(cmd, "Running structure refinement")

# If clustering is enabled (pclu=True)
if pclu:
    print("[DRfold2] ADDITIONAL STEP: CLUSTERING")
    print(f"[DRfold2] Running clustering process, output: {clufile}")
    
    # Run clustering process
    cmd = f'python {clupy} {ret_dir} {clufile}'
    run_cmd(cmd, "Running clustering")

    # Read clustering results
    lines = open(clufile).readlines()
    lines = [aline.strip() for aline in lines]
    lines = [aline for aline in lines if aline]
    
    cluster_count = len(lines) - 1
    print(f"[DRfold2] Found {cluster_count} additional clusters to process")

    # Process each cluster
    for i in range(1,len(lines)):
        print(f"[DRfold2] PROCESSING CLUSTER {i}/{cluster_count}")
        
        # Get return files for this cluster
        rets = lines[i].split()
        rets = [os.path.join(ret_dir,aret.replace('.pdb','.ret')) for aret in rets ]
        ret_str = ' '.join(rets)

        # Set up save prefixes for this cluster
        optsaveprefix =  os.path.join(cso_dir,f'opt_{str(i+1)}')
        save_prefix = os.path.join(cso_dir,f'sel_{str(i+1)}')
        
        print(f"[DRfold2] Cluster {i} Selection Process")
        print(f"[DRfold2] Found {len(rets)} return files for selection")
        print(f"[DRfold2] Selection output prefix: {save_prefix}")

        # Run selection process for this cluster
        cmd = f'python {selpython} {fastafile} {config_sel} {save_prefix} {ret_str}'
        run_cmd(cmd, f"Running selection for cluster {i}")
        
        print(f"[DRfold2] Cluster {i} Optimization Process")
        print(f"[DRfold2] Optimization output prefix: {optsaveprefix}")

        # Run optimization process for this cluster
        cmd = f'python {optpython} {fastafile} {optsaveprefix} {ret_dir} {save_prefix} {foldconfig}'
        run_cmd(cmd, f"Running optimization for cluster {i}")

        # Get the coarse-grained PDB and save refined structure for this cluster
        cgpdb = os.path.join(folddir,get_model_pdb(folddir,f'opt_{str(i+1)}'))
        savepdb = os.path.join(refdir,f'model_{str(i+1)}.pdb')
        
        print(f"[DRfold2] Cluster {i} Refinement Process")
        print(f"[DRfold2] Found optimized structure: {cgpdb}")
        print(f"[DRfold2] Final output will be saved to: {savepdb}")

        cmd = f'{arena} {cgpdb} {savepdb} 7'
        run_cmd(cmd, f"Running refinement for cluster {i}")

print("[DRfold2] PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
%%writefile /kaggle/working/DRfold2/PotentialFold/operations.py
"""
operations.py: Core Mathematical Operations for RNA Structure Analysis

This module provides essential mathematical operations for manipulating and analyzing
RNA 3D structures, organized into four main categories:

1. Basic Vector Operations:
   Functions for selecting coordinates and calculating distances between points,
   which form the foundation for all structural calculations.

2. Angle Calculations:
   Functions for computing bond angles and dihedral (torsion) angles between atoms,
   with differentiable implementations suitable for gradient-based optimization.

3. Rigid Body Transformations:
   Functions for determining optimal rotations and translations between sets of
   coordinates, enabling structure alignment and manipulation.

4. Sequence Utilities:
   Functions for converting RNA sequence data into standard 3D coordinate templates,
   allowing sequence-structure mapping.

These operations support the core functionality of RNA structure prediction, analysis,
and optimization throughout the codebase.
"""

import os
import torch
import torch.nn as nn
import numpy as np 
import math, sys, math
from io import BytesIO
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter
from subprocess import Popen, PIPE, STDOUT

# Use consistent epsilon value across all functions
EPS = 1e-8


# === Basic Vector Operations ===
def coor_selection(coor,mask):
    #[L,n,3],[L,n],byte
    return torch.masked_select(coor,mask.bool()).view(-1,3)

def pair_distance(x1, x2, eps=1e-6, p=2):
    # Use torch.cdist for p=2 (Euclidean) which is highly optimized
    if p == 2:
        return torch.cdist(x1, x2, p=2)
    
    # For other p-norms, avoid memory expansion with broadcasting
    x1_ = x1.unsqueeze(1)  # [n1, 1, dim]
    x2_ = x2.unsqueeze(0)  # [1, n2, dim]
    diff = torch.abs(x1_ - x2_)
    out = torch.pow(diff + eps, p).sum(dim=2)
    return torch.pow(out, 1. / p)


# === Angle Calculations ===
def angle(p0, p1, p2):
    # [b 3] 
    b0 = p0-p1
    b1 = p2-p1

    b0 = b0 / (torch.norm(b0, dim =-1, keepdim=True) + EPS)
    b1 = b1 / (torch.norm(b1, dim =-1, keepdim=True) + EPS)
    
    recos = torch.sum(b0*b1, -1)
    recos = torch.clamp(recos, -0.9999, 0.9999)
    return torch.acos(recos)

class torsion(Function):
    #PyTorch class to calculate differentiable torsion angle
    #https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    #https://salilab.org/modeller/manual/node492.html
    @staticmethod
    def forward(ctx, p0, p1, p2, p3):
        # Save input points for backward pass
        ctx.save_for_backward(p0, p1, p2, p3)

        # Calculate bond vectors
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2

        # Normalize the middle bond vector
        b1_norm = torch.norm(b1, dim=-1, keepdim=True) + 1e-8
        b1_unit = b1 / b1_norm

        # Project the other bonds onto the plane perpendicular to middle bond
        v = b0 - torch.sum(b0 * b1_unit, dim=-1, keepdim=True) * b1_unit
        w = b2 - torch.sum(b2 * b1_unit, dim=-1, keepdim=True) * b1_unit

        # Calculate torsion using the arctan2 formula (more stable than arccos)
        x = torch.sum(v * w, dim=-1)                                # cosine component
        y = torch.sum(torch.cross(b1_unit, v, dim=-1) * w, dim=-1)  # sine component

        return torch.atan2(y, x)

    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors from forward pass
        p0, p1, p2, p3 = ctx.saved_tensors

        # Calculate bond vectors
        r01 = p0 - p1
        r12 = p2 - p1
        r23 = p3 - p2

        # Calculate bond lengths with numerical stability
        d01 = torch.norm(r01, dim=-1, keepdim=True) + 1e-8
        d12 = torch.norm(r12, dim=-1, keepdim=True) + 1e-8
        d23 = torch.norm(r23, dim=-1, keepdim=True) + 1e-8

        # Normalize bond vectors
        e01 = r01 / d01
        e12 = r12 / d12
        e23 = r23 / d23

        # Calculate normal vectors to the two planes
        n1 = torch.cross(e01, e12, dim=-1)
        n2 = torch.cross(e12, e23, dim=-1)

        # Normalize normal vectors
        n1_norm = torch.norm(n1, dim=-1, keepdim=True) + 1e-8
        n2_norm = torch.norm(n2, dim=-1, keepdim=True) + 1e-8
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        # Calculate gradients for each atom
        # These are based on the analytical derivatives of dihedral angles
        g0 = torch.cross(e01, n1, dim=-1) / d01
        g1 = -g0 - torch.cross(e12, n1, dim=-1) / d12
        g2 = torch.cross(e12, n2, dim=-1) / d12 - torch.cross(e23, n2, dim=-1) / d23
        g3 = torch.cross(e23, n2, dim=-1) / d23

        # Apply chain rule with incoming gradient
        g0 = g0 * grad_output.unsqueeze(-1)
        g1 = g1 * grad_output.unsqueeze(-1)
        g2 = g2 * grad_output.unsqueeze(-1)
        g3 = g3 * grad_output.unsqueeze(-1)

        return g0, g1, g2, g3


def dihedral(input1, input2, input3, input4):
    return torsion.apply(input1, input2, input3, input4)



# === Rigid Body Transformations ===
def rigidFrom3Points(x):    
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    v1 = x3 - x2
    v2 = x1 - x2
    
    # Normalize v1 to get e1
    e1 = F.normalize(v1, p=2, dim=-1)
    
    # Project v2 onto e1 and subtract to get the component orthogonal to e1
    u2 = v2 - e1 * (torch.einsum('bn,bn->b', e1, v2)[:, None])
    
    # Normalize u2 to get e2
    e2 = F.normalize(u2, p=2, dim=-1)
    
    # Cross product to get e3
    e3 = torch.cross(e1, e2, dim=-1)
    
    return torch.stack([e1, e2, e3], dim=1)


# return the direction from to_q to from_p
def Kabsch_rigid(bases,x1,x2,x3):
    # Early return for empty input
    if x1.shape[0] == 0:
        return torch.empty(0, 3, 3), torch.empty(0, 3)
    
    the_dim=1
    to_q = torch.stack([x1,x2,x3],dim=the_dim)
    biasq=torch.mean(to_q,dim=the_dim,keepdim=True)
    q=to_q-biasq
    m = torch.einsum('bnz,bny->bzy',bases,q)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r,biasq.squeeze()



# === Sequence Utilities ===
def Get_base(seq,basenpy_standard):
    base_num = basenpy_standard.shape[1]
    basenpy = np.zeros([len(seq),base_num,3])
    seqnpy = np.array(list(seq))
    basenpy[seqnpy=='A']=basenpy_standard[0]
    basenpy[seqnpy=='a']=basenpy_standard[0]

    basenpy[seqnpy=='G']=basenpy_standard[1]
    basenpy[seqnpy=='g']=basenpy_standard[1]

    basenpy[seqnpy=='C']=basenpy_standard[2]
    basenpy[seqnpy=='c']=basenpy_standard[2]

    basenpy[seqnpy=='U']=basenpy_standard[3]
    basenpy[seqnpy=='u']=basenpy_standard[3]

    basenpy[seqnpy=='T']=basenpy_standard[3]
    basenpy[seqnpy=='t']=basenpy_standard[3]
    
    return torch.from_numpy(basenpy).double()
%%writefile /kaggle/working/DRfold2/PotentialFold/Optimization.py
#! /nfs/amino-home/liyangum/miniconda3/bin/python
import torch
import random
import numpy as np 
import os, json, sys

import Cubic, Potential
import operations
import a2b, rigid
import torch.optim as opt
from scipy.optimize import minimize
import pickle

torch.manual_seed(6)
np.random.seed(9)
random.seed(9)


Scale_factor = 1.0
USEGEO = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def readconfig(configfile=''):
    config=[]
    expdir=os.path.dirname(os.path.abspath(__file__))
    if configfile=='':
        configfile=os.path.join(expdir,'lib','ddf.json')
    config=json.load(open(configfile,'r'))
    return config 

    
class Structure:
    def __init__(self,fastafile,geofiles,saveprefix,initial_ret,foldconfig):
        self.config=readconfig(foldconfig)
        self.seqfile=fastafile
        self.init_ret = initial_ret
        self.foldconfig = foldconfig
        self.geofiles = geofiles
        self.rets = [pickle.load(open(refile,'rb')) for refile  in geofiles]
        self.txs=[]
        for ret in self.rets:
            self.txs.append( torch.from_numpy(ret['coor']).double().to(device))
        self.handle_geo()
        self.pair = []
        for ret in self.rets:
            self.pair.append(torch.from_numpy(ret['plddt']).double().to(device))
        self.saveprefix=saveprefix
        self.seq=open(fastafile).readlines()[1].strip()
        self.L=len(self.seq)
        basenpy = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','base.npy'))
        self.basex = operations.Get_base(self.seq,basenpy).to(device)
        othernpy = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','other2.npy'))
        self.otherx = operations.Get_base(self.seq,othernpy).to(device)
        sidenpy = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','side.npy'))
        self.sidex = operations.Get_base(self.seq,sidenpy).to(device)
        
        self.init_mask()
        self.init_paras()
        self._init_fape()
        self.tx2ds = [td.to(device) for td in self.tx2ds]
        self.local_weight = torch.ones(self.L,self.L).to(device)
        
        for i in range(self.L):
            for j in range(i+1,min(self.L,i+2)):
                self.local_weight[i,j] = self.local_weight[j,i] = 4
            for j in range(i+2,min(self.L,i+3)):
                self.local_weight[i,j] = self.local_weight[j,i] = 3
            for j in range(i+3,min(self.L,i+4)):
                self.local_weight[i,j] = self.local_weight[j,i] = 2

    def _init_fape(self):
        self.tx2ds = []
        for tx in self.txs:
            true_rot,true_trans = operations.Kabsch_rigid(self.basex,tx[:,0],tx[:,1],tx[:,2])
            true_x2 = tx[:,None,:,:] - true_trans[None,:,None,:]
            true_x2 = torch.einsum('ijnd,jde->ijne',true_x2,true_rot.transpose(-1,-2))
            self.tx2ds.append(true_x2)
    
    def handle_geo(self):
        oldkeys=['dist_p','dist_c','dist_n']
        newkeys=['pp','cc','nn']
        self.geos=[]
        for ret in self.rets:
            geo = {}
            for nk,ok in zip(newkeys,oldkeys):
                geo[nk] = torch.from_numpy(ret[ok].astype(np.float64)).to(device) + 0
            self.geos.append(geo)


    def init_mask(self):
        halfmask=np.zeros([self.L,self.L])
        fullmask=np.zeros([self.L,self.L])
        for i in range(self.L):
            for j in range(i+1,self.L):
                halfmask[i,j]=1
                fullmask[i,j]=1
                fullmask[j,i]=1
        self.halfmask=(torch.DoubleTensor(halfmask) > 0.5).to(device)
        self.fullmask=(torch.DoubleTensor(fullmask) > 0.5).to(device)
        self.clash_mask = torch.zeros([self.L,self.L,22,22], device=device)
        for i in range(self.L):
            for j in range(i+1,self.L):
                self.clash_mask[i,j]=1

        for i in range(self.L):
             self.clash_mask[i,i,:6,7:]=1

        for i in range(self.L-1):
            self.clash_mask[i,i+1,:,0]=0
            self.clash_mask[i,i+1,0,:]=0
            self.clash_mask[i,i+1,:,5]=0
            self.clash_mask[i,i+1,5,:]=0

        self.side_mask = rigid.side_mask(self.seq).to(device)
        self.side_mask = (self.side_mask[:,None,:,None] * self.side_mask[None,:,None,:]).to(device)
        self.clash_mask = ((self.clash_mask > 0.5) * (self.side_mask > 0.5)).to(device)

        self.geo_confimask_cc = []
        self.geo_confimask_pp = []
        self.geo_confimask_nn = []
        for geo in self.geos:
            confimask_cc = geo['cc'][:,:,-1] < 0.5
            confimask_pp = geo['pp'][:,:,-1] < 0.5
            confimask_nn = geo['nn'][:,:,-1] < 0.5
            self.geo_confimask_cc.append(confimask_cc)
            self.geo_confimask_pp.append(confimask_pp)
            self.geo_confimask_nn.append(confimask_nn)


    def init_paras(self):
        self.geo_cc = []
        self.geo_pp = []
        self.geo_nn = []
        self.cs_coefs = {'cc': [], 'pp': [], 'nn': []}
        self.cs_knots = {'cc': [], 'pp': [], 'nn': []}
        for geo in self.geos:
            cc_cs,cc_decs=Cubic.dis_cubic(geo['cc'],2,40,36)
            pp_cs,pp_decs=Cubic.dis_cubic(geo['pp'],2,40,36)
            nn_cs,nn_decs=Cubic.dis_cubic(geo['nn'],2,40,36)
            self.geo_cc.append([cc_cs,cc_decs])
            self.geo_pp.append([pp_cs,pp_decs])
            self.geo_nn.append([nn_cs,nn_decs])
            
            L = self.L
            cc_coefs_np  = np.stack([[cc_cs[i,j].c for j in range(L)] for i in range(L)], axis=0)
            cc_knots_np  = np.stack([[cc_cs[i,j].x for j in range(L)] for i in range(L)], axis=0)
            self.cs_coefs['cc'].append(torch.from_numpy(cc_coefs_np).to(device))
            self.cs_knots['cc'].append(torch.from_numpy(cc_knots_np).to(device))
            
            pp_coefs_np  = np.stack([[pp_cs[i,j].c for j in range(L)] for i in range(L)], axis=0)
            pp_knots_np  = np.stack([[pp_cs[i,j].x for j in range(L)] for i in range(L)], axis=0)
            self.cs_coefs['pp'].append(torch.from_numpy(pp_coefs_np).to(device))
            self.cs_knots['pp'].append(torch.from_numpy(pp_knots_np).to(device))
            
            nn_coefs_np  = np.stack([[nn_cs[i,j].c for j in range(L)] for i in range(L)], axis=0)
            nn_knots_np  = np.stack([[nn_cs[i,j].x for j in range(L)] for i in range(L)], axis=0)
            self.cs_coefs['nn'].append(torch.from_numpy(nn_coefs_np).to(device))
            self.cs_knots['nn'].append(torch.from_numpy(nn_knots_np).to(device))


    def compute_bb_clash(self,coor,other_coor):
        com_coor = torch.cat([coor,other_coor],dim=1)
        com_dis  = (com_coor[:,None,:,None,:] - com_coor[None,:,None,:,:]).norm(dim=-1)
        dynamicmask2_vdw= (com_dis <= 3.15) * (self.clash_mask)
        vdw_dynamic = Potential.LJpotential(com_dis[dynamicmask2_vdw],3.15)
        return vdw_dynamic.sum()*self.config['weight_vdw']

    def compute_full_clash(self,coor,other_coor,side_coor):
        com_coor = torch.cat([coor[:,:2],other_coor,side_coor],dim=1)
        com_dis  = (com_coor[:,None,:,None,:] - com_coor[None,:,None,:,:]).norm(dim=-1)
        dynamicmask2_vdw= (com_dis <= 2.5) * (self.clash_mask)
        vdw_dynamic = Potential.LJpotential(com_dis[dynamicmask2_vdw],2.5)
        return vdw_dynamic.sum()*self.config['weight_vdw']


    def _cubic_pair_energy(self, atom_map, geo_cs, geo_confimask, weight_key):
        """General cubic-spline energy for CC/PP/NN pairs."""
        min_dis, max_dis, bin_num = 2, 40, 36
        dev = atom_map.device
        upper_th = max_dis - ((max_dis - min_dis) / bin_num) * 0.5
        lower_th = 2.5
        total = torch.zeros((), device=dev, dtype=torch.double)
        spline_key   = weight_key.split('_')[1]  # 'cc', 'pp', or 'nn'
        coeffs_list  = self.cs_coefs[spline_key]
        knots_list   = self.cs_knots[spline_key]
        for block_idx, mask_block in enumerate(geo_confimask):
            mask = (atom_map <= upper_th) & mask_block & self.fullmask & (atom_map >= lower_th)
            idx = mask.nonzero(as_tuple=True)
            if idx[0].numel() > 1:
                coef  = coeffs_list[block_idx][idx]
                knots = knots_list[block_idx][idx]
                part1 = Potential.cubic_distance(atom_map[mask], coef, knots, min_dis, max_dis, bin_num).sum() * self.config[weight_key] * 0.5
            else:
                part1 = torch.zeros((), device=dev)
            part2 = ((atom_map <= lower_th) & mask_block & self.fullmask).sum() * self.config[weight_key]
            total = total + part1 + part2
        return total

    def compute_cc_energy(self, coor):
        atom_map = operations.pair_distance(coor[:,1], coor[:,1])
        return self._cubic_pair_energy(atom_map, self.geo_cc, self.geo_confimask_cc, 'weight_cc')
    
    def compute_pp_energy(self, coor):
        atom_map = operations.pair_distance(coor[:,0], coor[:,0])
        return self._cubic_pair_energy(atom_map, self.geo_pp, self.geo_confimask_pp, 'weight_pp')
    
    def compute_nn_energy(self, coor):
        atom_map = operations.pair_distance(coor[:,-1], coor[:,-1])
        return self._cubic_pair_energy(atom_map, self.geo_nn, self.geo_confimask_nn, 'weight_nn')

    def compute_pccp_energy(self,coor):
        p_atoms=coor[:,0]
        c_atoms=coor[:,1]
        pccpmap=operations.dihedral( p_atoms[self.pccpi], c_atoms[self.pccpi], c_atoms[self.pccpj] ,p_atoms[self.pccpj]                  )
        neg_log = Potential.cubic_torsion(pccpmap,self.pccp_coe,self.pccp_x,36)
        return neg_log.sum()*self.config['weight_pccp']

    def compute_cnnc_energy(self,coor):
        n_atoms=coor[:,-1]
        c_atoms=coor[:,1]
        pccpmap=operations.dihedral( c_atoms[self.cnnci], n_atoms[self.cnnci], n_atoms[self.cnncj] ,c_atoms[self.cnncj]                  )
        neg_log = Potential.cubic_torsion(pccpmap,self.cnnc_coe,self.cnnc_x,36)
        return neg_log.sum()*self.config['weight_cnnc']

    def compute_pnnp_energy(self,coor):
        n_atoms=coor[:,-1]
        p_atoms=coor[:,0]
        pccpmap=operations.dihedral( p_atoms[self.pnnpi], n_atoms[self.pnnpi], n_atoms[self.pnnpj] ,p_atoms[self.pnnpj]                  )
        neg_log = Potential.cubic_torsion(pccpmap,self.pnnp_coe,self.pnnp_x,36)
        return neg_log.sum()*self.config['weight_pnnp']

    def compute_pcc_energy(self,coor):
        p_atoms=coor[:,1]
        c_atoms=coor[:,2]
        pccmap=operations.angle( p_atoms[self.pcci], c_atoms[self.pcci], c_atoms[self.pccj]                   )
        neg_log = Potential.cubic_angle(pccmap,self.pcc_coe,self.pcc_x,12)
        return neg_log.sum()*self.config['weight_pcc']

    def compute_fape_energy(self,coor,ep=1e-3,epmax=20):
        energy= 0
        for tx in self.tx2ds:
            px_mean = coor[:,[1]]
            p_rot   = operations.rigidFrom3Points(coor)
            p_tran  = px_mean[:,0]
            pred_x2 = coor[:,None,:,:] - p_tran[None,:,None,:] # Lx Lrot N , 3
            pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse
            errmap=torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep )
            energy = energy + torch.sum(  torch.clamp(errmap,max=epmax)        )
        return energy * self.config['weight_fape']

    def compute_bond_energy(self,coor,other_coor):
        # 3.87
        o3 = other_coor[:-1,-2]
        p  = coor[1:,0]
        dis = (o3-p).norm(dim=-1)
        energy = ((dis-1.607)**2).sum()
        return energy * self.config['weight_bond']

    def tooth_func(self,errmap, ep = 0.05):
        return -1/(errmap/10+ep) + (1/ep)

    def reweight_func(self,ww):
        reweighting = torch.pow(ww,self.config['pair_weight_power'])
        reweighting[ww < self.config['pair_weight_min']] = 0
        return reweighting

    def compute_fape_energy_fromquat(self,x,coor,ep=1e-6,epmax=100):
        energy= 0
        p_rot,px_mean = a2b.Non2rot(x[:,:9],x.shape[0]),x[:,9:]
        pred_x2 = coor[:,None,:,:] - px_mean[None,:,None,:] # Lx Lrot N , 3
        pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse
        for tx,weightplddt in zip(self.tx2ds,self.pair):

            tamplate_dist_map = torch.min( tx.norm(dim=-1), dim=2   )[0]
            errmap=torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep ) 
            energy = energy + torch.sum( ( (torch.clamp(errmap,max=self.config['FAPE_max'])**self.config['pair_error_power'])  * self.reweight_func(weightplddt[...,None]) * self.local_weight[...,None] )[tamplate_dist_map>self.config['pair_rest_min_dist']]    )

        return energy * self.config['weight_fape']


    def energy(self,rama):
        coor=a2b.quat2b(self.basex,rama[:,9:])
        other_coor = a2b.quat2b(self.otherx,rama[:,9:])
        side_coor = a2b.quat2b(self.sidex,torch.cat([rama[:,:9],coor[:,-1]],dim=-1))
        
        if self.config['weight_cc']>0:
            E_cc= self.compute_cc_energy(coor) / len(self.rets)
        else:
            E_cc=0
        if self.config['weight_pp']>0:
            E_pp= self.compute_pp_energy(coor) / len(self.rets)
        else:
            E_pp=0
        if self.config['weight_nn']>0:
            E_nn= self.compute_nn_energy(coor) / len(self.rets)
        else:
            E_nn=0

        if self.config['weight_pccp']>0:
            E_pccp= self.compute_pccp_energy(coor) / len(self.rets)
        else:
            E_pccp=0

        if self.config['weight_cnnc']>0:
            E_cnnc= self.compute_cnnc_energy(coor)  / len(self.rets)
        else:
            E_cnnc=0

        if self.config['weight_pnnp']>0:
            E_pnnp= self.compute_pnnp_energy(coor) / len(self.rets)
        else:
            E_pnnp=0

        if self.config['weight_vdw']>0:
            E_vdw= self.compute_full_clash(coor,other_coor,side_coor)
        else:
            E_vdw=0

        if self.config['weight_fape']>0:
            E_fape= self.compute_fape_energy_fromquat(rama[:,9:],coor) / len(self.rets)
        else:
            E_fape=0
        if self.config['weight_bond']>0:
            E_bond= self.compute_bond_energy(coor,other_coor)
        else:
            E_bond=0
        return  E_vdw + E_fape + E_bond + E_pp + E_cc + E_nn + E_pccp + E_cnnc + E_pnnp


    def obj_func_grad_np(self,rama_):
        rama=torch.DoubleTensor(rama_)
        rama.requires_grad=True
        if rama.grad:
            rama.grad.zero_()
        f=self.energy(rama.view(self.L,21))*Scale_factor
        grad_value=autograd.grad(f,rama)[0]
        return grad_value.data.numpy().astype(np.float64)
    
    def obj_func_np(self,rama_):
        rama=torch.DoubleTensor(rama_)
        rama=rama.view(self.L,21)
        with torch.no_grad():
            f=self.energy(rama)*Scale_factor
            return f.item()


    def foldning(self):
        ilter = self.init_ret
        # 1) get initial quaternions (double precision)
        try:
            init_q = self.init_quat(ilter).double()
        except:
            init_q = self.init_quat_safe(ilter).double()

        # 2) move to target device (GPU if available), enable grad
        param = init_q.to(device).clone().detach().requires_grad_(True)

        # 3) set up PyTorch LBFGS optimizer over `param`
        optimizer = opt.LBFGS(
            [param],
            max_iter=self.config.get('max_iter', 300),
            tolerance_grad=1e-6,
            tolerance_change=1e-9,
            history_size=10,
            line_search_fn='strong_wolfe'
        )

        # 4) define the “closure” that LBFGS will call to reevaluate loss + gradients
        def closure():
            optimizer.zero_grad()                                 # clear old grads
            E = self.energy(param.view(self.L,21)) * Scale_factor # compute ∂E/∂param
            E.backward()
            return E

        # 5) run LBFGS until convergence (it calls closure repeatedly)
        optimizer.step(closure)

        # 6) write out final PDB
        final_energy = self.energy(param.view(self.L,21)).item()
        self.outpdb(param, self.saveprefix + '.pdb', energystr=str(final_energy))


    def outpdb(self,rama,savefile,start=0,end=10000,energystr=''):
        # bring baseframes and quaternion data onto CPU to prevent device mismatch
        basex_cpu = self.basex.detach().cpu()
        otherx_cpu = self.otherx.detach().cpu()
        sidex_cpu = self.sidex.detach().cpu()
        shaped_rama = rama.view(self.L,21).detach().cpu()
        # compute backbone and other coords
        coor_np = a2b.quat2b(basex_cpu, shaped_rama[:,9:]).detach().cpu().numpy()
        other_np = a2b.quat2b(otherx_cpu, shaped_rama[:,9:]).detach().cpu().numpy()
        coor = torch.FloatTensor(coor_np)
        # compute side atom coords
        side_coor_NP = a2b.quat2b(sidex_cpu, torch.cat([shaped_rama[:,:9], coor[:,-1]], dim=-1)).detach().cpu().numpy()
        
        Atom_name=[' P  '," C4'",' N1 ']
        Other_Atom_name = [" O5'"," C5'"," C3'"," O3'"," C1'"]
        other_last_name = ['O',"C","C","O","C"]

        side_atoms=         [' N1 ',' C2 ',' O2 ',' N2 ',' N3 ',' N4 ',' C4 ',' O4 ',' C5 ',' C6 ',' O6 ',' N6 ',' N7 ',' N8 ',' N9 ']
        side_last_name =    ['N',      "C",   "O",   "N",   "N",   'N',   'C',   'O',   'C',   'C',   'O',   'N',    'N', 'N','N']

        base_dict = rigid.base_table()
        last_name=['P','C','N']
        wstr=[f'REMARK {str(energystr)}']
        templet='%6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s'
        count=1
        for i in range(self.L):
            if self.seq[i] in ['a','g','A','G']:
                Atom_name = [' P  '," C4'",' N9 ']
                #atoms = ['P','C4']

            elif self.seq[i] in ['c','u','C','U']:
                Atom_name = [' P  '," C4'",' N1 ']
            for j in range(coor_np.shape[1]):
                outs=('ATOM  ',count,Atom_name[j],self.seq[i],'A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],0,0,last_name[j],'')
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                    count+=1

            for j in range(other_np.shape[1]):
                outs=('ATOM  ',count,Other_Atom_name[j],self.seq[i],'A',i+1,other_np[i][j][0],other_np[i][j][1],other_np[i][j][2],0,0,other_last_name[j],'')
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                    count+=1
            
        wstr='\n'.join(wstr)
        wfile=open(savefile,'w')
        wfile.write(wstr)
        wfile.close()
    
    def outpdb_coor(self,coor_np,savefile,start=0,end=1000,energystr=''):
        Atom_name=[' P  '," C4'",' N1 ']
        last_name=['P','C','N']
        wstr=[f'REMARK {str(energystr)}']
        templet='%6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s'
        count=1
        for i in range(self.L):
            if self.seq[i] in ['a','g','A','G']:
                Atom_name = [' P  '," C4'",' N9 ']

            elif self.seq[i] in ['c','u','C','U']:
                Atom_name = [' P  '," C4'",' N1 ']
            for j in range(coor_np.shape[1]):
                outs=('ATOM  ',count,Atom_name[j],self.seq[i],'A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],0,0,last_name[j],'')
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                count+=1
            
        wstr='\n'.join(wstr)
        wfile=open(savefile,'w')
        wfile.write(wstr)
        wfile.close()


    def init_quat(self,ii):
        x = torch.rand([self.L,21])
        x[:,18:] = self.txs[ii].mean(dim=1)
        init_coor = self.txs[ii]
        biasq = torch.mean(init_coor,dim=1,keepdim=True)
        q = init_coor - biasq
        m = torch.einsum('bnz,bny->bzy',self.basex,q).reshape([self.L,-1])
        x[:,:9] = x[:,9:18] = m
        x.requires_grad_()
        return x

    def init_quat_safe(self,ii):
        x = torch.rand([self.L,21])
        x[:,18:] = self.txs[ii].mean(dim=1)
        init_coor = self.txs[ii]
        biasq = torch.mean(init_coor,dim=1,keepdim=True)
        q = init_coor - biasq + torch.rand([self.L,3,3])
        m = (torch.einsum('bnz,bny->bzy',self.basex,q) + torch.eye(3)[None,:,:]).reshape([self.L,-1])
        x[:,:9] = x[:,9:18] = m
        x.requires_grad_()
        return x


if __name__ == '__main__': 

    fastafile=sys.argv[1]
    saveprefix=sys.argv[2]
    retdirs  =sys.argv[3]
    ret_score = sys.argv[4]
    foldconfig = sys.argv[5]

    savepare = os.path.dirname(saveprefix)
    if not os.path.isdir(savepare):
        os.makedirs(savepare)

    num_of_models = readconfig(foldconfig)['num_of_models']

    score_dict = readconfig(ret_score)
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1])
    lowest_n_keys = [item[0] for item in sorted_items][:num_of_models]
    bestkey = lowest_n_keys[0] + ''
    print("Before sort:", lowest_n_keys)
    lowest_n_keys.sort()
    print("After sort:", lowest_n_keys)
    bestindex = lowest_n_keys.index(bestkey)

    current_ret = bestkey
    retfiles = [os.path.join(retdirs, afile) for afile in lowest_n_keys]
    stru = Structure(fastafile, retfiles, saveprefix + '_from_' + current_ret, bestindex, foldconfig)
    stru.foldning()
%%writefile /kaggle/working/DRfold2/PotentialFold/Selection.py
#! /nfs/amino-home/liyangum/miniconda3/bin/python
import numpy
import torch
import torch.autograd as autograd
import numpy as np 

import random
import Cubic, Potential
import operations
import os, json, sys

import a2b, rigid
import torch.optim as opt
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from scipy.optimize import fmin_l_bfgs_b,fmin_cg,fmin_bfgs
from scipy.optimize import minimize
import lbfgs_rosetta
import pickle
import shutil

torch.manual_seed(6)
torch.set_num_threads(4)
np.random.seed(9)
random.seed(9)

Scale_factor = 1.0
USEGEO = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def readconfig(configfile=''):
    config=[]
    expdir=os.path.dirname(os.path.abspath(__file__))
    if configfile=='':
        configfile=os.path.join(expdir,'lib','ddf.json')
    config=json.load(open(configfile,'r'))
    return config 

    
class Structure:
    def __init__(self, fastafile, geofiles, foldconfig, saveprefix):
        # Load Configuration and Inputs
        self.config = readconfig(foldconfig)
        self.seqfile = fastafile
        self.foldconfig = foldconfig
        self.geofiles = geofiles

        # Load Model Results
        self.rets = [pickle.load(open(refile, 'rb')) for refile  in geofiles]
        
        # Extract Coordinates
        self.txs = []
        for ret in self.rets:
            self.txs.append(torch.from_numpy(ret['coor']).double().to(device))
        
        # Handle Geometrical Data
        self.handle_geo()

        # Extract pLDDT Scores
        self.pair = []
        for ret in self.rets:
            self.pair.append( torch.from_numpy(ret['plddt']).double().to(device))
        
        # Store Output and Sequence Info
        self.saveprefix = saveprefix
        self.seq = open(fastafile).readlines()[1].strip()
        self.L = len(self.seq)
        
        # Load Reference Arrays for Structure Construction
        basenpy = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', 'base.npy'))
        self.basex = operations.Get_base(self.seq, basenpy).double().to(device)
        
        othernpy = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', 'other2.npy'))
        self.otherx = operations.Get_base(self.seq, othernpy).double().to(device)
        
        sidenpy = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', 'side.npy'))
        self.sidex = operations.Get_base(self.seq, sidenpy).double().to(device)        
        
        # Initialize Masks, Parameters, and FAPE
        self.init_mask()
        self.init_paras()
        self._init_fape()
    

    def _init_fape(self):
        self.tx2ds = []
        for tx in self.txs:
            true_rot, true_trans = operations.Kabsch_rigid(self.basex, tx[:, 0], tx[:, 1], tx[:, 2])
            true_x2 = tx[:, None, :, :] - true_trans[None, :, None, :]
            true_x2 = torch.einsum('ijnd,jde->ijne', true_x2, true_rot.transpose(-1,-2))
            self.tx2ds.append(true_x2)
    

    def handle_geo(self):
        oldkeys = ['dist_p', 'dist_c', 'dist_n']
        newkeys = ['pp', 'cc', 'nn']
        self.geos = []
        geo = {'pp':0, 'cc':0, 'nn':0}
        
        for ret in self.rets:    
            for nk, ok in zip(newkeys, oldkeys):
                geo[nk] = geo[nk] + (ret[ok].astype(np.float64) /(len(self.rets)))
        self.geos.append(geo)


    def init_mask(self):
        halfmask=np.zeros([self.L,self.L])
        fullmask=np.zeros([self.L,self.L])
        for i in range(self.L):
            for j in range(i+1,self.L):
                halfmask[i,j]=1
                fullmask[i,j]=1
                fullmask[j,i]=1
        self.halfmask=torch.DoubleTensor(halfmask) > 0.5
        self.fullmask=torch.DoubleTensor(fullmask) > 0.5
        self.clash_mask = torch.zeros([self.L,self.L,22,22])
        for i in range(self.L):
            for j in range(i+1,self.L):
                self.clash_mask[i,j]=1

        for i in range(self.L):
             self.clash_mask[i,i,:6,7:]=1

        for i in range(self.L-1):
            self.clash_mask[i,i+1,:,0]=0
            self.clash_mask[i,i+1,0,:]=0
            self.clash_mask[i,i+1,:,5]=0
            self.clash_mask[i,i+1,5,:]=0

        self.side_mask = rigid.side_mask(self.seq)
        self.side_mask = self.side_mask[:,None,:,None] * self.side_mask[None,:,None,:]
        self.clash_mask = (self.clash_mask > 0.5) * (self.side_mask > 0.5)

        self.geo_confimask_cc = []
        self.geo_confimask_pp = []
        self.geo_confimask_nn = []
        for geo in self.geos:
            confimask_cc = torch.DoubleTensor(geo['cc'][:,:,-1]) < 0.5
            confimask_pp = torch.DoubleTensor(geo['pp'][:,:,-1]) < 0.5
            confimask_nn = torch.DoubleTensor(geo['nn'][:,:,-1]) < 0.5
            self.geo_confimask_cc.append(confimask_cc)
            self.geo_confimask_pp.append(confimask_pp)
            self.geo_confimask_nn.append(confimask_nn)

        # Move masks and confimasks to the GPU/CPU device
        self.halfmask = self.halfmask.to(device)
        self.fullmask = self.fullmask.to(device)
        self.clash_mask = self.clash_mask.to(device)
        self.side_mask = self.side_mask.to(device)
        # geo_confimasks are lists
        self.geo_confimask_cc = [m.to(device) for m in self.geo_confimask_cc]
        self.geo_confimask_pp = [m.to(device) for m in self.geo_confimask_pp]
        self.geo_confimask_nn = [m.to(device) for m in self.geo_confimask_nn]


    def init_paras(self):
        self.geo_cc = []
        self.geo_pp = []
        self.geo_nn = []
        self.cs_coefs = {'cc': [], 'pp': [], 'nn': []}
        self.cs_knots = {'cc': [], 'pp': [], 'nn': []}
        for geo in self.geos:
            cc_cs, cc_decs = Cubic.dis_cubic(geo['cc'], 2, 40, 36)
            pp_cs, pp_decs = Cubic.dis_cubic(geo['pp'], 2, 40, 36)
            nn_cs, nn_decs = Cubic.dis_cubic(geo['nn'], 2, 40, 36)
            self.geo_cc.append([cc_cs, cc_decs])
            self.geo_pp.append([pp_cs, pp_decs])
            self.geo_nn.append([nn_cs, nn_decs])
            L = self.L
            cc_coefs_np = np.stack([[cc_cs[i,j].c for j in range(L)] for i in range(L)], axis=0)
            cc_knots_np = np.stack([[cc_cs[i,j].x for j in range(L)] for i in range(L)], axis=0)
            self.cs_coefs['cc'].append(torch.from_numpy(cc_coefs_np).to(device))
            self.cs_knots['cc'].append(torch.from_numpy(cc_knots_np).to(device))
            pp_coefs_np = np.stack([[pp_cs[i,j].c for j in range(L)] for i in range(L)], axis=0)
            pp_knots_np = np.stack([[pp_cs[i,j].x for j in range(L)] for i in range(L)], axis=0)
            self.cs_coefs['pp'].append(torch.from_numpy(pp_coefs_np).to(device))
            self.cs_knots['pp'].append(torch.from_numpy(pp_knots_np).to(device))
            nn_coefs_np = np.stack([[nn_cs[i,j].c for j in range(L)] for i in range(L)], axis=0)
            nn_knots_np = np.stack([[nn_cs[i,j].x for j in range(L)] for i in range(L)], axis=0)
            self.cs_coefs['nn'].append(torch.from_numpy(nn_coefs_np).to(device))
            self.cs_knots['nn'].append(torch.from_numpy(nn_knots_np).to(device))
     

    def _cubic_pair_energy(self, atom_map, geo_cs, geo_confimask, weight_key):
        """General cubic-spline energy for CC/PP/NN pairs."""
        min_dis, max_dis, bin_num = 2, 40, 36
        dev = atom_map.device
        upper_th = max_dis - ((max_dis - min_dis) / bin_num) * 0.5
        lower_th = 2.5
        total = torch.zeros((), device=dev, dtype=torch.double)
        spline_key = weight_key.split('_')[1]
        coeffs_list = self.cs_coefs[spline_key]
        knots_list = self.cs_knots[spline_key]
        for block_idx, mask_block in enumerate(geo_confimask):
            mask = (atom_map <= upper_th) & mask_block & self.fullmask & (atom_map >= lower_th)
            idx = mask.nonzero(as_tuple=True)
            if idx[0].numel() > 1:
                coef = coeffs_list[block_idx][idx]
                knots = knots_list[block_idx][idx]
                part1 = Potential.cubic_distance(atom_map[mask], coef, knots, min_dis, max_dis, bin_num).sum() * self.config[weight_key] * 0.5
            else:
                part1 = torch.zeros((), device=dev, dtype=torch.double)
            part2 = ((atom_map <= lower_th) & mask_block & self.fullmask).sum() * self.config[weight_key]
            total = total + part1 + part2
        return total

    # GPU-friendly torsion and angle energy helpers
    def _cubic_torsion_energy(self, atom_map, coef, x_vals, weight_key, num_bin):
        energy = Potential.cubic_torsion(atom_map, coef, x_vals, num_bin)
        return energy.sum() * self.config[weight_key]

    def _cubic_angle_energy(self, atom_map, coef, x_vals, weight_key, num_bin):
        energy = Potential.cubic_angle(atom_map, coef, x_vals, num_bin)
        return energy.sum() * self.config[weight_key]

    def compute_cc_energy(self, coor):
        atom_map = operations.pair_distance(coor[:,1], coor[:,1])
        return self._cubic_pair_energy(atom_map, self.geo_cc, self.geo_confimask_cc, 'weight_cc')

    def compute_pp_energy(self, coor):
        atom_map = operations.pair_distance(coor[:,0], coor[:,0])
        return self._cubic_pair_energy(atom_map, self.geo_pp, self.geo_confimask_pp, 'weight_pp')

    def compute_nn_energy(self, coor):
        atom_map = operations.pair_distance(coor[:,-1], coor[:,-1])
        return self._cubic_pair_energy(atom_map, self.geo_nn, self.geo_confimask_nn, 'weight_nn')

    def compute_pccp_energy(self, coor):
        # P-C-C-P dihedral energy on GPU
        p = coor[:, 0]
        c = coor[:, 1]
        dia = operations.dihedral(
            p[self.pccpi], c[self.pccpi], c[self.pccpj], p[self.pccpj]
        )
        return self._cubic_torsion_energy(dia, self.pccp_coe, self.pccp_x, 'weight_pccp', 36)

    def compute_cnnc_energy(self, coor):
        # C-N-N-C dihedral energy on GPU
        n = coor[:, -1]
        c = coor[:, 1]
        dia = operations.dihedral(
            c[self.cnnci], n[self.cnnci], n[self.cnncj], c[self.cnncj]
        )
        return self._cubic_torsion_energy(dia, self.cnnc_coe, self.cnnc_x, 'weight_cnnc', 36)

    def compute_pnnp_energy(self, coor):
        # P-N-N-P dihedral energy on GPU
        n = coor[:, -1]
        p = coor[:, 0]
        dia = operations.dihedral(
            p[self.pnnpi], n[self.pnnpi], n[self.pnnpj], p[self.pnnpj]
        )
        return self._cubic_torsion_energy(dia, self.pnnp_coe, self.pnnp_x, 'weight_pnnp', 36)

    def compute_pcc_energy(self, coor):
        # P-C-C angle energy on GPU
        p = coor[:, 1]
        c = coor[:, 2]
        ang = operations.angle(
            p[self.pcci], c[self.pcci], c[self.pccj]
        )
        return self._cubic_angle_energy(ang, self.pcc_coe, self.pcc_x, 'weight_pcc', 12)

    def compute_fape_energy(self,coor,ep=1e-3,epmax=20):
        energy= 0
        for tx in self.tx2ds:
            px_mean = coor[:,[1]]
            p_rot   = operations.rigidFrom3Points(coor)
            p_tran  = px_mean[:,0]
            pred_x2 = coor[:,None,:,:] - p_tran[None,:,None,:] # Lx Lrot N , 3
            pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse
            errmap=torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep )
            energy = energy + torch.sum(  torch.clamp(errmap,max=epmax)        )
        return energy * self.config['weight_fape']

    def compute_bond_energy(self,coor,other_coor):
        # 3.87
        o3 = other_coor[:-1,-2]
        p  = coor[1:,0]
        dis = (o3-p).norm(dim=-1)
        energy = ((dis-1.607)**2).sum()
        return energy * self.config['weight_bond']

    def tooth_func(self,errmap, ep = 0.05):
        return -1/(errmap/10+ep) + (1/ep)
    
    def reweight_func(self,ww):
        reweighting = torch.pow(ww,self.config['pair_weight_power'])
        reweighting[ww < self.config['pair_weight_min']] = 0
        return reweighting
    
    def compute_fape_energy_fromquat(self,x,coor,ep=1e-6,epmax=100):
        energy= 0
        p_rot,px_mean = a2b.Non2rot(x[:,:9],x.shape[0]),x[:,9:]
        pred_x2 = coor[:,None,:,:] - px_mean[None,:,None,:] # Lx Lrot N , 3
        pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse

        for tx,weightplddt in zip(self.tx2ds,self.pair):
            tamplate_dist_map = torch.min( tx.norm(dim=-1), dim=2   )[0]
            errmap=torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep ) 
            energy = energy + torch.sum( ( (torch.clamp(errmap,max=self.config['FAPE_max'])**self.config['pair_error_power'])  * self.reweight_func(weightplddt[...,None]) )[tamplate_dist_map>self.config['pair_rest_min_dist']]    )

        return energy * self.config['weight_fape']
    
    def compute_fape_energy_fromcoor(self,coor,ep=1e-6,epmax=100):
        energy= 0
        
        p_rot,px_mean = operations.Kabsch_rigid(self.basex,coor[:,0],coor[:,1],coor[:,2])
        pred_x2 = coor[:,None,:,:] - px_mean[None,:,None,:] # Lx Lrot N , 3
        pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse
        
        for tx,weightplddt in zip(self.tx2ds,self.pair):
            tamplate_dist_map = torch.min( tx.norm(dim=-1), dim=2   )[0]
            errmap=torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep ) 
            energy = energy + torch.sum( ( (torch.clamp(errmap,max=self.config['FAPE_max'])**self.config['pair_error_power'])  * self.reweight_func(weightplddt[...,None]) )[tamplate_dist_map>self.config['pair_rest_min_dist']]    )

        return energy * self.config['weight_fape']
    
    
    def energy(self, rama):
        coor = a2b.quat2b(self.basex, rama[:, 9:])
        other_coor = a2b.quat2b(self.otherx, rama[:, 9:])
        side_coor = a2b.quat2b(self.sidex, torch.cat([rama[:, :9], coor[:, -1]], dim=-1))

        E_cc = self.compute_cc_energy(coor) / len(self.geofiles) if self.config['weight_cc'] > 0 else 0
        E_pp = self.compute_pp_energy(coor) / len(self.geofiles) if self.config['weight_pp'] > 0 else 0
        E_nn = self.compute_nn_energy(coor) / len(self.geofiles) if self.config['weight_nn'] > 0 else 0
        E_pccp = self.compute_pccp_energy(coor) / len(self.geofiles) if self.config['weight_pccp'] > 0 else 0
        E_cnnc = self.compute_cnnc_energy(coor) / len(self.geofiles) if self.config['weight_cnnc'] > 0 else 0
        E_pnnp = self.compute_pnnp_energy(coor) / len(self.geofiles) if self.config['weight_pnnp'] > 0 else 0
        E_vdw = self.compute_full_clash(coor, other_coor, side_coor) if self.config['weight_vdw'] > 0 else 0
        E_fape = self.compute_fape_energy_fromquat(rama[:, 9:], coor) / len(self.geofiles) if self.config['weight_fape'] > 0 else 0
        E_bond = self.compute_bond_energy(coor, other_coor) if self.config['weight_bond'] > 0 else 0

        return E_vdw + E_fape + E_bond + E_pp + E_cc + E_nn + E_pccp + E_cnnc + E_pnnp


    def energy_from_coor(self, coor):
        E_cc = self.compute_cc_energy(coor) if self.config['weight_cc'] > 0 else 0
        E_pp = self.compute_pp_energy(coor) if self.config['weight_pp'] > 0 else 0
        E_nn = self.compute_nn_energy(coor) if self.config['weight_nn'] > 0 else 0
        E_fape = (self.compute_fape_energy_fromcoor(coor) / len(self.geofiles)) if self.config['weight_fape'] > 0 else 0
        print(E_fape, E_pp, E_cc, E_nn)
        return E_fape + E_pp + E_cc + E_nn 

    def obj_func_grad_np(self,rama_):
        rama=torch.DoubleTensor(rama_)
        rama.requires_grad=True
        if rama.grad:
            rama.grad.zero_()
        f=self.energy(rama.view(self.L,21))*Scale_factor
        grad_value=autograd.grad(f,rama)[0]
        return grad_value.data.numpy().astype(np.float64)
    
    def obj_func_np(self,rama_):
        rama=torch.DoubleTensor(rama_)
        rama=rama.view(self.L,21)
        with torch.no_grad():
            f = self.energy(rama)*Scale_factor
            return f.item()

    def saveconfig(self,dict,confile):
        json_object = json.dumps(dict, indent = 4)
        wfile = open(confile,'w')
        wfile.write(json_object)
        wfile.close()
    
    def scoring(self):
        geoscale = self.config['geo_scale']
        self.config['weight_pp'] = geoscale * self.config['weight_pp']
        self.config['weight_cc'] = geoscale * self.config['weight_cc']
        self.config['weight_nn'] = geoscale * self.config['weight_nn']
        self.config['weight_pccp'] = geoscale * self.config['weight_pccp']
        self.config['weight_cnnc'] = geoscale * self.config['weight_cnnc']
        self.config['weight_pnnp'] = geoscale * self.config['weight_pnnp']  
        
        energy_dict = {}
        saveenergy_dict  = {}
        
        with torch.no_grad():
            for retfile, tx in zip(self.geofiles, self.txs):
                one = self.energy_from_coor(tx)
                aaretfile = os.path.basename(retfile) 
                energy_dict[aaretfile] = one.item()
                saveenergy_dict[retfile] = one.item()
            self.saveconfig(energy_dict, self.saveprefix)


    def foldning(self):
        minenergy=1e16
        count=0
        for tx in self.txs:
            count+=1
        
        minirama=None

        ilter = self.init_ret
        selected_ret = self.geofiles[ilter]
        try:
            rama=self.init_quat(ilter).data.numpy()
            self.config=readconfig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','vdw.json'))
            rama = fmin_l_bfgs_b(func=self.obj_func_np, x0=rama,  fprime=self.obj_func_grad_np,iprint=10,maxfun=100)[0]
            rama = rama.flatten()
        except:
            rama=self.init_quat_safe(ilter).data.numpy()
            self.config=readconfig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','vdw.json'))
            rama = fmin_l_bfgs_b(func=self.obj_func_np, x0=rama,  fprime=self.obj_func_grad_np,iprint=10,maxfun=100)[0]
            rama = rama.flatten()
            
        self.config=readconfig(self.foldconfig)
        geoscale = self.config['geo_scale']
        self.config['weight_pp'] =geoscale * self.config['weight_pp']
        self.config['weight_cc'] =geoscale * self.config['weight_cc']
        self.config['weight_nn'] =geoscale * self.config['weight_nn']
        self.config['weight_pccp'] =geoscale * self.config['weight_pccp']
        self.config['weight_cnnc'] =geoscale * self.config['weight_cnnc']
        self.config['weight_pnnp'] =geoscale * self.config['weight_pnnp']
        for i in range(3):
            line_min = lbfgs_rosetta.ArmijoLineMinimization(self.obj_func_np,self.obj_func_grad_np,True,len(rama),120)
            lbfgs_opt = lbfgs_rosetta.lbfgs(self.obj_func_np,self.obj_func_grad_np)
            rama=lbfgs_opt.run(rama,256,lbfgs_rosetta.absolute_converge_test,line_min,8000,self.obj_func_np,self.obj_func_grad_np,1e-9)
        newrama=rama+0.0
        newrama=torch.DoubleTensor(newrama) 
        current_energy =self.obj_func_np(rama)

        if current_energy < minenergy:
            print(current_energy,minenergy)
            minenergy=current_energy
            self.outpdb(newrama,self.saveprefix+'.pdb',energystr=str(current_energy))


    def outpdb(self,rama,savefile,start=0,end=10000,energystr=''):
        coor_np=a2b.quat2b(self.basex,rama.view(self.L,21)[:,9:]).data.numpy()
        other_np=a2b.quat2b(self.otherx,rama.view(self.L,21)[:,9:]).data.numpy()
        shaped_rama=rama.view(self.L,21)
        coor = torch.FloatTensor(coor_np)
        side_coor_NP = a2b.quat2b(self.sidex,torch.cat([shaped_rama[:,:9],coor[:,-1]],dim=-1)).data.numpy()
        
        Atom_name=[' P  '," C4'",' N1 ']
        Other_Atom_name = [" O5'"," C5'"," C3'"," O3'"," C1'"]
        other_last_name = ['O',"C","C","O","C"]

        side_atoms=         [' N1 ',' C2 ',' O2 ',' N2 ',' N3 ',' N4 ',' C4 ',' O4 ',' C5 ',' C6 ',' O6 ',' N6 ',' N7 ',' N8 ',' N9 ']
        side_last_name =    ['N',      "C",   "O",   "N",   "N",   'N',   'C',   'O',   'C',   'C',   'O',   'N',    'N', 'N','N']

        base_dict = rigid.base_table()
        
        last_name=['P','C','N']
        wstr=[f'REMARK {str(energystr)}']
        templet='%6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s'
        count=1
        for i in range(self.L):
            if self.seq[i] in ['a','g','A','G']:
                Atom_name = [' P  '," C4'",' N9 ']

            elif self.seq[i] in ['c','u','C','U']:
                Atom_name = [' P  '," C4'",' N1 ']
            for j in range(coor_np.shape[1]):
                outs=('ATOM  ',count,Atom_name[j],self.seq[i],'A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],0,0,last_name[j],'')
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                    count+=1

            for j in range(other_np.shape[1]):
                outs=('ATOM  ',count,Other_Atom_name[j],self.seq[i],'A',i+1,other_np[i][j][0],other_np[i][j][1],other_np[i][j][2],0,0,other_last_name[j],'')
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                    count+=1
            
        wstr='\n'.join(wstr)
        wfile=open(savefile,'w')
        wfile.write(wstr)
        wfile.close()
    
    
    def outpdb_coor(self,coor_np,savefile,start=0,end=1000,energystr=''):
        Atom_name=[' P  '," C4'",' N1 ']
        last_name=['P','C','N']
        wstr=[f'REMARK {str(energystr)}']
        templet='%6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s'
        count=1
        for i in range(self.L):
            if self.seq[i] in ['a','g','A','G']:
                Atom_name = [' P  '," C4'",' N9 ']

            elif self.seq[i] in ['c','u','C','U']:
                Atom_name = [' P  '," C4'",' N1 ']
            
            for j in range(coor_np.shape[1]):
                outs=('ATOM  ',count,Atom_name[j],self.seq[i],'A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],0,0,last_name[j],'')
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                count+=1
            
        wstr='\n'.join(wstr)
        wfile=open(savefile,'w')
        wfile.write(wstr)
        wfile.close()


if __name__ == '__main__': 

    fastafile = sys.argv[1]
    foldconfig = sys.argv[2]
    save_prefix = sys.argv[3]
    retfiles = sys.argv[4:]

    save_parent_dir = os.path.dirname(save_prefix)
    if not os.path.isdir(save_parent_dir):
        os.makedirs(save_parent_dir)

    retfiles.sort()
    print(retfiles)

    stru = Structure(fastafile, retfiles, foldconfig, save_prefix)    
    stru.scoring()
# %%writefile /kaggle/working/DRfold2/cfg_97/test_modeldir.py
# import random
# random.seed(0)
# import numpy as np
# np.random.seed(0)
# import os,sys,re,random
# from numpy import select
# import torch
# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# expdir=os.path.dirname(os.path.abspath(__file__))


# import torch.optim as opt
# from torch.nn import functional as F
# import data,util
# import EvoMSA2XYZ,basic
# import math
# import pickle
# Batch_size=3
# Num_cycle=3
# TEST_STEP=1000
# VISION_STEP=50
# device = sys.argv[1]


# expdir=os.path.dirname(os.path.abspath(__file__))
# expround=expdir.split('_')[-1]
# model_path=os.path.join(expdir,'others','models')

# testdir=os.path.join(expdir,'others','preds')
# basenpy_standard= np.load( os.path.join(os.path.dirname(os.path.abspath(__file__)),'base.npy'  )  )

# def data_collect(pdb_seq):
#     aa_type = data.parse_seq(pdb_seq)
#     base = data.Get_base(pdb_seq,basenpy_standard)
#     seq_idx = np.arange(len(pdb_seq)) + 1

#     msa=aa_type[None,:]
#     msa=torch.from_numpy(msa).to(device)
#     msa=torch.cat([msa,msa],0)
#     msa=F.one_hot(msa.long(),6).float()

#     base_x = torch.from_numpy(base).float().to(device)
#     seq_idx = torch.from_numpy(seq_idx).long().to(device)
#     return msa,base_x,seq_idx
#     predxs,plddts = model.pred(msa,seq_idx,ss,base_x,sample_1['alpha_0'])



# def classifier(infasta,out_prefix,model_dir):
#     with torch.no_grad():
#         lines = open(infasta).readlines()[1:]
#         seqs = [aline.strip() for aline in lines]
#         seq = ''.join(seqs)
#         msa,base_x,seq_idx = data_collect(seq)
        
#         msa_dim=6+1
#         m_dim,s_dim,z_dim = 64,64,64
#         N_ensemble, N_cycle = 6, 12
#         model=EvoMSA2XYZ.MSA2XYZ(msa_dim-1,msa_dim,N_ensemble,N_cycle,m_dim,s_dim,z_dim)
#         model.to(device)
#         model.eval()
#         models = os.listdir(  model_dir   )
#         models = [amodel for amodel in models if 'model' in amodel and 'opt' not in amodel]

#         models.sort()

#         for amodel in models:
#             saved_model = os.path.join(model_dir, amodel)
#             model.load_state_dict(torch.load(saved_model, map_location='cpu'), strict=True)
#             ret = model.pred(msa, seq_idx, None, base_x, np.array(list(seq)))

#             util.outpdb(ret['coor'], seq_idx, seq, out_prefix+f'{amodel}.pdb')
#             pickle.dump(ret, open(out_prefix+f'{amodel}.ret', 'wb'))


# if __name__ == '__main__':
#     infasta, out_prefix, model_dir = sys.argv[2], sys.argv[3], sys.argv[4]
#     classifier(infasta, out_prefix, model_dir)
%%writefile /kaggle/working/DRfold2/PotentialFold/Cubic.py
import numpy as np 
from scipy.interpolate import CubicSpline,UnivariateSpline
import os
from torch.autograd import Function
import torch
import math

def fit_dis_cubic(dis_matrix,min_dis,max_dis,num_bin):
    # convert torch Tensor on GPU to numpy array for SciPy
    if isinstance(dis_matrix, torch.Tensor):
        dis_matrix = dis_matrix.detach().cpu().numpy()
    dis_region=np.zeros(num_bin)
    for i in range(num_bin):
        dis_region[i]=min_dis+(i+0.5)*(max_dis-min_dis)*1.0/num_bin
    L=dis_matrix.shape[0]
    csnp=[]
    decsnp=[]
    for i in range(L):
        css=[]
        decss=[]
        for j in range(L):
            y=-np.log(      (dis_matrix[i,j,1:-1]+1e-8) / (dis_matrix[i,j,[-2]]+1e-8)              )
            x=dis_region
            x[0]=-0.0001
            y[0]= max(10,y[1]+4)
            cs= CubicSpline(x,y)
            decs=cs.derivative()
            css.append(cs)
            decss.append(decs)
        csnp.append(css)
        decsnp.append(decss)
    return np.array(csnp),np.array(decsnp)

def dis_cubic(out,min_dis,max_dis,num_bin):
    print('fitting cubic distance')
    cs,decs=fit_dis_cubic(out,min_dis,max_dis,num_bin)
    return cs,decs



def cubic_matrix_torsion(dis_matrix,min_dis,max_dis,num_bin):
    dis_region=np.zeros(num_bin)
    bin_size=(max_dis-min_dis)/num_bin
    for i in range(num_bin):
        dis_region[i]=min_dis+(i+0.5)*(max_dis-min_dis)*1.0/num_bin
    L=dis_matrix.shape[0]
    csnp=[]
    decsnp=[]
    for i in range(L):
        css=[]
        decss=[]
        for j in range(L):
            y=-np.log(      dis_matrix[i,j,:-1]+1e-8             )
            x=dis_region
            x=np.append(x,x[-1]+bin_size)
            y=np.append(y,y[0])
            cs= CubicSpline(x,y,bc_type='periodic')
            decs=cs.derivative()
            css.append(cs)
            decss.append(decs)
        csnp.append(css)
        decsnp.append(decss)
    return np.array(csnp),np.array(decsnp)
def torsion_cubic(out,min_dis,max_dis,num_bin):
    print('fitting cubic')
    cs,decs=cubic_matrix_torsion(out,min_dis,max_dis,num_bin)
    return cs,decs

def cubic_matrix_angle(dis_matrix,min_dis,max_dis,num_bin): # 0 - np.pi 12
    dis_region=np.zeros(num_bin)
    bin_size=(max_dis-min_dis)/num_bin
    for i in range(num_bin):
        dis_region[i]=min_dis+(i+0.5)*(max_dis-min_dis)*1.0/num_bin
    L=dis_matrix.shape[0]
    csnp=[]
    decsnp=[]
    for i in range(L):
        css=[]
        decss=[]
        for j in range(L):
            y=-np.log(      dis_matrix[i,j,:-1]+1e-8             )
            x=dis_region

            x=np.concatenate([[x[0]-bin_size*3,x[0]-bin_size*2,x[0]-bin_size], x,[x[-1]+bin_size,x[-1]+bin_size*2,x[-1]+bin_size*3]               ])
            y=np.concatenate([ [y[2],y[1],y[0]],y,[y[-1],y[-2],y[-3]]                                                                                                                    ])

            cs= CubicSpline(x,y)
            decs=cs.derivative()

            css.append(cs)
            decss.append(decs)
        csnp.append(css)
        decsnp.append(decss)

    return np.array(csnp),np.array(decsnp)
def angle_cubic(out,min_dis,max_dis,num_bin):

    print('fitting angle cubic')
    cs,decs=cubic_matrix_angle(out,min_dis,max_dis,num_bin)

    return cs,decs
# Define an improved function to run DRfold2 that captures output
def predict_rna_structures_drfold2(sequence, target_id):
    """Use DRfold2 to predict RNA structures with proper output capture"""
    import subprocess
    from subprocess import PIPE, STDOUT
    
    # Create FASTA file for this sequence
    fasta_path = os.path.join(fasta_dir, f"{target_id}.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">{target_id}\n{sequence}\n")
    
    # Run DRfold2 with proper output capture
    output_dir = os.path.join(predictions_dir, target_id)
    cmd = f"python /kaggle/working/DRfold2/DRfold_infer.py {fasta_path} {output_dir} 1"
    
    print(f"Running command: {cmd}")
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=PIPE, 
        stderr=STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Print output in real-time
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            print(line)
    
    # Get return code and check success
    return_code = process.wait()
    if return_code != 0:
        print(f"DRfold2 failed with return code {return_code}")
        return None
    
    # Clean up FASTA file to save space
    os.remove(fasta_path)
    
    # Extract coordinates
    relax_dir = os.path.join(output_dir, "relax")
    if not os.path.isdir(relax_dir):
        print(f"Warning: No relax directory found for {target_id}")
        relax_dir = output_dir
    
    # Get up to 5 PDB files
    pdb_files = sorted([f for f in os.listdir(relax_dir) if f.endswith(".pdb")])[:5]
    
    if not pdb_files:
        print(f"Warning: No PDB files found for {target_id}")
        # Return None to indicate failure
        return None
    
    # Parse PDB files to extract C1' coordinates
    predictions = []
    for pdb_file in pdb_files:
        file_path = os.path.join(relax_dir, pdb_file)
        
        # Read PDB file
        coords = []
        with open(file_path, "r") as f:
            residue_map = {}
            for line in f:
                if line.startswith("ATOM") and " C1' " in line:
                    parts = line.split()
                    resid = int(parts[5])  # Residue ID as integer
                    x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
                    residue_map[resid] = (x, y, z)
            
            # Ensure we have coordinates for all residues
            for j in range(1, len(sequence) + 1):
                if j in residue_map:
                    coords.append(residue_map[j])
                else:
                    # If residue not found, use zeros
                    print(f"Warning: Residue {j} not found in {pdb_file} for {target_id}")
                    coords.append((0.0, 0.0, 0.0))
        
        predictions.append(coords)
    
    # Clean up PDB files to save space
    if is_submission_mode:
        shutil.rmtree(output_dir)
    
    # If we have fewer than 5 predictions, duplicate the last one
    while len(predictions) < 5:
        predictions.append(predictions[-1] if predictions else [(0.0, 0.0, 0.0) for _ in range(len(sequence))])
    
    return predictions[:5]  # Return exactly 5 predictions
# Vectorized version of process_labels function
def process_labels_vectorized(labels_df):
    # Extract target_id from ID column (remove last part after underscore)
    labels_df = labels_df.copy()
    labels_df['target_id'] = labels_df['ID'].str.rsplit('_', n=1).str[0]
    
    # Sort by target_id and resid for proper ordering
    labels_df = labels_df.sort_values(['target_id', 'resid'])
    
    # Group by target_id and convert coordinates to arrays
    coords_dict = {}
    for target_id, group in labels_df.groupby('target_id'):
        # Extract coordinates as numpy array in one operation
        coords_dict[target_id] = group[['x_1', 'y_1', 'z_1']].values
    
    return coords_dict

def find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, top_n=5):
    similar_seqs = []
    query_seq_obj = Seq(query_seq)

    for _, row in train_seqs_df.iterrows():
        target_id = row['target_id']
        train_seq = row['sequence']

        # Skip if coordinates not available
        if target_id not in train_coords_dict:
            continue

        # Skip if sequence is too different in length (more than 40% difference)
        if abs(len(train_seq) - len(query_seq)) / max(len(train_seq), len(query_seq)) > 0.4:
            continue

        # Perform sequence alignment
        alignments = pairwise2.align.globalms(query_seq_obj, train_seq, 2.9, -1, -10, -0.5, one_alignment_only=True)

        if alignments:
            alignment = alignments[0]
            similarity_score = alignment.score / (2 * min(len(query_seq), len(train_seq)))
            similar_seqs.append((target_id, train_seq, similarity_score, train_coords_dict[target_id]))

    # Sort by similarity score (higher is better) and return top N
    similar_seqs.sort(key=lambda x: x[2], reverse=True)
    return similar_seqs[:top_n]


def adaptive_rna_constraints(coordinates, sequence, confidence=1.0):
    # Make a copy of coordinates to refine
    refined_coords = coordinates.copy()
    n_residues = len(sequence)
    
    # Calculate constraint strength (inverse of confidence)
    # High confidence templates receive gentler constraints
    constraint_strength = 0.8 * (1.0 - min(confidence, 0.8))
    
    # 1. Sequential distance constraints (consecutive nucleotides)
    # More flexible distance range (statistical distribution from PDB)
    seq_min_dist = 5.5  # Minimum sequential distance
    seq_max_dist = 6.5  # Maximum sequential distance
    
    for i in range(n_residues - 1):
        current_pos = refined_coords[i]
        next_pos = refined_coords[i+1]
        
        # Calculate current distance
        current_dist = np.linalg.norm(next_pos - current_pos)
        
        # Only adjust if significantly outside expected range
        if current_dist < seq_min_dist or current_dist > seq_max_dist:
            # Calculate target distance (midpoint of range)
            target_dist = (seq_min_dist + seq_max_dist) / 2
            
            # Get direction vector
            direction = next_pos - current_pos
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            
            # Apply partial adjustment based on constraint strength
            adjustment = (target_dist - current_dist) * constraint_strength
            
            # Only adjust the next position to preserve the overall fold
            refined_coords[i+1] = current_pos + direction * (current_dist + adjustment)
    
    # 2. Steric clash prevention (more conservative)
    min_allowed_distance = 3.8  # Minimum distance between non-consecutive C1' atoms
    
    # Calculate all pairwise distances
    dist_matrix = distance_matrix(refined_coords, refined_coords)
    
    # Find severe clashes (atoms too close)
    severe_clashes = np.where((dist_matrix < min_allowed_distance) & (dist_matrix > 0))
    
    # Fix severe clashes
    for idx in range(len(severe_clashes[0])):
        i, j = severe_clashes[0][idx], severe_clashes[1][idx]
        
        # Skip consecutive nucleotides and previously processed pairs
        if abs(i - j) <= 1 or i >= j:
            continue
            
        # Get current positions and distance
        pos_i = refined_coords[i]
        pos_j = refined_coords[j]
        current_dist = dist_matrix[i, j]
        
        # Calculate necessary adjustment but scale by constraint strength
        direction = pos_j - pos_i
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Calculate partial adjustment
        adjustment = (min_allowed_distance - current_dist) * constraint_strength
        
        # Move points apart
        refined_coords[i] = pos_i - direction * (adjustment / 2)
        refined_coords[j] = pos_j + direction * (adjustment / 2)
    
    # 3. Very light base-pair constraining (if confidence is low)
    if constraint_strength > 0.3:  # Only apply if template confidence is low
        # Simple Watson-Crick base pairs
        pairs = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        
        # Scan for potential base pairs
        for i in range(n_residues):
            base_i = sequence[i]
            complement = pairs.get(base_i)
            
            if not complement:
                continue
                
            # Look for complementary bases within a reasonable range
            for j in range(i + 3, min(i + 20, n_residues)):
                if sequence[j] == complement:
                    # Calculate current distance
                    current_dist = np.linalg.norm(refined_coords[i] - refined_coords[j])
                    
                    # Only consider if distance suggests potential pairing
                    if 8.0 < current_dist < 14.0:
                        # Target 10.5Å as generic base-pair C1'-C1' distance
                        target_dist = 10.5
                        
                        # Calculate very gentle adjustment (scaled by constraint_strength)
                        adjustment = (target_dist - current_dist) * (constraint_strength * 0.3)
                        
                        # Get direction vector
                        direction = refined_coords[j] - refined_coords[i]
                        direction = direction / (np.linalg.norm(direction) + 1e-10)
                        
                        # Apply very gentle adjustment to both positions
                        refined_coords[i] = refined_coords[i] - direction * (adjustment / 2)
                        refined_coords[j] = refined_coords[j] + direction * (adjustment / 2)
                        
                        # Only consider one potential pair per base (closest match)
                        break
    
    return refined_coords

def adapt_template_to_query(query_seq, template_seq, template_coords, alignment=None):
    if alignment is None:
        from Bio.Seq import Seq
        from Bio import pairwise2
        
        query_seq_obj = Seq(query_seq)
        template_seq_obj = Seq(template_seq)
        alignments = pairwise2.align.globalms(query_seq_obj, template_seq_obj, 2.9, -1, -10, -0.5, one_alignment_only=True)
        
        if not alignments:
            return generate_improved_rna_structure(query_seq)
            
        alignment = alignments[0]
    
    aligned_query = alignment.seqA
    aligned_template = alignment.seqB
    
    query_coords = np.zeros((len(query_seq), 3))
    query_coords.fill(np.nan)
    
    # Map template coordinates to query
    query_idx = 0
    template_idx = 0
    
    for i in range(len(aligned_query)):
        query_char = aligned_query[i]
        template_char = aligned_template[i]
        
        if query_char != '-' and template_char != '-':
            if template_idx < len(template_coords):
                query_coords[query_idx] = template_coords[template_idx]
            template_idx += 1
            query_idx += 1
        elif query_char != '-' and template_char == '-':
            query_idx += 1
        elif query_char == '-' and template_char != '-':
            template_idx += 1
    
    # IMPROVED GAP FILLING - maintains RNA backbone geometry
    backbone_distance = 5.9  # Typical C1'-C1' distance
    
    # Fill gaps by maintaining realistic backbone connectivity
    for i in range(len(query_coords)):
        if np.isnan(query_coords[i, 0]):
            # Find nearest valid neighbors
            prev_valid = next_valid = None
            
            for j in range(i-1, -1, -1):
                if not np.isnan(query_coords[j, 0]):
                    prev_valid = j
                    break
                    
            for j in range(i+1, len(query_coords)):
                if not np.isnan(query_coords[j, 0]):
                    next_valid = j
                    break
            
            if prev_valid is not None and next_valid is not None:
                # Interpolate along realistic RNA backbone path
                gap_size = next_valid - prev_valid
                total_distance = np.linalg.norm(query_coords[next_valid] - query_coords[prev_valid])
                expected_distance = gap_size * backbone_distance
                
                # If gap is compressed, extend it realistically
                if total_distance < expected_distance * 0.7:
                    direction = query_coords[next_valid] - query_coords[prev_valid]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                    
                    # Place intermediate points along extended path
                    for k, idx in enumerate(range(prev_valid + 1, next_valid)):
                        progress = (k + 1) / gap_size
                        base_pos = query_coords[prev_valid] + direction * expected_distance * progress
                        
                        # Add slight curvature for realism
                        perpendicular = np.cross(direction, [0, 0, 1])
                        if np.linalg.norm(perpendicular) < 1e-6:
                            perpendicular = np.cross(direction, [1, 0, 0])
                        perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-10)
                        
                        curve_amplitude = 2.0 * np.sin(progress * np.pi)
                        query_coords[idx] = base_pos + perpendicular * curve_amplitude
                else:
                    # Linear interpolation for normal gaps
                    for k, idx in enumerate(range(prev_valid + 1, next_valid)):
                        weight = (k + 1) / gap_size
                        query_coords[idx] = (1 - weight) * query_coords[prev_valid] + weight * query_coords[next_valid]
            
            elif prev_valid is not None:
                # Extend from previous position
                if prev_valid > 0 and not np.isnan(query_coords[prev_valid-1, 0]):
                    direction = query_coords[prev_valid] - query_coords[prev_valid-1]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                else:
                    direction = np.array([1.0, 0.0, 0.0])
                
                steps_needed = i - prev_valid
                for step in range(1, steps_needed + 1):
                    pos_idx = prev_valid + step
                    if pos_idx < len(query_coords):
                        query_coords[pos_idx] = query_coords[prev_valid] + direction * backbone_distance * step
            
            elif next_valid is not None:
                # Work backwards from next position
                direction = np.array([-1.0, 0.0, 0.0])  # Default backward direction
                steps_needed = next_valid - i
                for step in range(steps_needed, 0, -1):
                    pos_idx = next_valid - step
                    if pos_idx >= 0:
                        query_coords[pos_idx] = query_coords[next_valid] - direction * backbone_distance * step
    
    # Final cleanup
    query_coords = np.nan_to_num(query_coords)
    return query_coords




def generate_improved_rna_structure(sequence):
    """
    Generate a more realistic RNA structure fallback based on sequence patterns
    and basic RNA structure principles.
    
    Args:
        sequence: RNA sequence string
        
    Returns:
        Array of 3D coordinates
    """
    n_residues = len(sequence)
    coordinates = np.zeros((n_residues, 3))
    
    # Analyze sequence to predict structural elements
    # Look for complementary regions that could form base pairs
    potential_stems = identify_potential_stems(sequence)
    
    # Default parameters
    radius_helix = 10.0
    radius_loop = 15.0
    rise_per_residue_helix = 2.5
    rise_per_residue_loop = 1.5
    angle_per_residue_helix = 0.6
    angle_per_residue_loop = 0.3
    
    # Assign structural classifications
    structure_types = assign_structure_types(sequence, potential_stems)
    
    # Generate coordinates based on predicted structure
    current_pos = np.array([0.0, 0.0, 0.0])
    current_direction = np.array([0.0, 0.0, 1.0])
    current_angle = 0.0
    
    for i in range(n_residues):
        if structure_types[i] == 'stem':
            # Part of a helical stem
            current_angle += angle_per_residue_helix
            coordinates[i] = [
                radius_helix * np.cos(current_angle), 
                radius_helix * np.sin(current_angle), 
                current_pos[2] + rise_per_residue_helix
            ]
            current_pos = coordinates[i]
        elif structure_types[i] == 'loop':
            # Part of a loop
            current_angle += angle_per_residue_loop
            z_shift = rise_per_residue_loop * np.sin(current_angle * 0.5)
            coordinates[i] = [
                radius_loop * np.cos(current_angle), 
                radius_loop * np.sin(current_angle), 
                current_pos[2] + z_shift
            ]
            current_pos = coordinates[i]
        else:
            # Single-stranded region
            # Add some randomness to make it look more realistic
            jitter = np.random.normal(0, 1, 3) * 2.0
            coordinates[i] = current_pos + jitter
            current_pos = coordinates[i]
            
    return coordinates

def identify_potential_stems(sequence):
    """
    Identify potential stem regions by looking for self-complementary segments.
    
    Args:
        sequence: RNA sequence string
        
    Returns:
        List of tuples (start1, end1, start2, end2) representing potentially paired regions
    """
    complementary_bases = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    min_stem_length = 3
    potential_stems = []
    
    # Simple stem identification
    for i in range(len(sequence) - min_stem_length):
        for j in range(i + min_stem_length + 3, len(sequence) - min_stem_length + 1):
            # Check if regions could form a stem
            potential_stem_len = min(min_stem_length, len(sequence) - j)
            is_stem = True
            
            for k in range(potential_stem_len):
                if sequence[i+k] not in complementary_bases or \
                   complementary_bases[sequence[i+k]] != sequence[j+potential_stem_len-k-1]:
                    is_stem = False
                    break
            
            if is_stem:
                potential_stems.append((i, i+potential_stem_len-1, j, j+potential_stem_len-1))
    
    return potential_stems

def assign_structure_types(sequence, potential_stems):
    """
    Assign each nucleotide to a structural element type.
    
    Args:
        sequence: RNA sequence string
        potential_stems: List of tuples representing stem regions
        
    Returns:
        List of structure types ('stem', 'loop', 'single')
    """
    structure_types = ['single'] * len(sequence)
    
    # Mark stem regions
    for stem in potential_stems:
        start1, end1, start2, end2 = stem
        for i in range(end1 - start1 + 1):
            structure_types[start1 + i] = 'stem'
            structure_types[end2 - i] = 'stem'
    
    # Mark loop regions (regions between paired regions)
    for i in range(len(potential_stems) - 1):
        _, end1, start2, _ = potential_stems[i]
        next_start1, _, _, _ = potential_stems[i+1]
        
        if next_start1 > end1 + 1 and start2 > next_start1:
            for j in range(end1 + 1, next_start1):
                structure_types[j] = 'loop'
    
    return structure_types



# Function to create a more realistic RNA structure when no good templates are found
def generate_rna_structure(sequence, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    n_residues = len(sequence)
    coordinates = np.zeros((n_residues, 3))
    
    # Initialize the first few residues in a helix
    for i in range(min(3, n_residues)):
        angle = i * 0.6
        coordinates[i] = [10.0 * np.cos(angle), 10.0 * np.sin(angle), i * 2.5]
    
    # Add more complex folding patterns
    current_direction = np.array([0.0, 0.0, 1.0])  # Start moving along z-axis
    
    # Define base-pairing tendencies (G-C and A-U pairs)
    for i in range(3, n_residues):
        # Check for potential base-pairing in the sequence
        has_pair = False
        pair_idx = -1
        
        # Simple detection of complementary bases (G-C, A-U)
        complementary = {'G': 'C', 'C': 'G', 'A': 'U', 'U': 'A'}
        current_base = sequence[i]
        
        # Look for potential base-pairing within a window before the current position
        window_size = min(i, 15)  # Look back up to 15 bases
        for j in range(i-window_size, i):
            if j >= 0 and sequence[j] == complementary.get(current_base, 'X'):
                # Found a potential pair
                has_pair = True
                pair_idx = j
                break
        
        if has_pair and i - pair_idx <= 10 and random.random() < 0.7:
            # Try to create a base-pair by positioning this nucleotide near its pair
            pair_pos = coordinates[pair_idx]
            
            # Create a position that's roughly opposite to the pair
            random_offset = np.random.normal(0, 1, 3) * 2.0
            base_pair_distance = 10.0 + random.uniform(-1.0, 1.0)
            
            # Calculate a vector from base-pair toward center of structure
            center = np.mean(coordinates[:i], axis=0)
            direction = center - pair_pos
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            
            # Position new nucleotide in the general direction of the "center"
            coordinates[i] = pair_pos + direction * base_pair_distance + random_offset
            
            # Update direction for next nucleotide
            current_direction = np.random.normal(0, 0.3, 3)
            current_direction = current_direction / (np.linalg.norm(current_direction) + 1e-10)
            
        else:
            # No base-pairing detected, continue with the current fold direction
            # Randomly rotate current direction to simulate RNA flexibility
            if random.random() < 0.3:
                # More significant direction change
                angle = random.uniform(0.2, 0.6)
                axis = np.random.normal(0, 1, 3)
                axis = axis / (np.linalg.norm(axis) + 1e-10)
                rotation = R.from_rotvec(angle * axis)
                current_direction = rotation.apply(current_direction)
            else:
                # Small random changes in direction
                current_direction += np.random.normal(0, 0.15, 3)
                current_direction = current_direction / (np.linalg.norm(current_direction) + 1e-10)
            
            # Distance between consecutive nucleotides (3.5-4.5Å is typical)
            step_size = random.uniform(3.5, 4.5)
            
            # Update position
            coordinates[i] = coordinates[i-1] + step_size * current_direction
    
    return coordinates


def predict_rna_structures(sequence, target_id, train_seqs_df, train_coords_dict, n_predictions=5):
    predictions = []
    
    # Find similar sequences in the training data
    similar_seqs = find_similar_sequences(sequence, train_seqs_df, train_coords_dict, top_n=n_predictions)
    
    # If we found any similar sequences, use them as templates
    if similar_seqs:
        for i, (template_id, template_seq, similarity_score, template_coords) in enumerate(similar_seqs):
            # Adapt template coordinates to the query sequence
            adapted_coords = adapt_template_to_query(sequence, template_seq, template_coords)
            
            if adapted_coords is not None:
                # Apply adaptive constraints based on template similarity
                # For high similarity templates, apply very gentle constraints
                refined_coords = adaptive_rna_constraints(adapted_coords, sequence, confidence=similarity_score)
                
                # Add some randomness (less for better templates)
                random_scale = max(0.05, 0.8 - similarity_score)  # Reduced randomness
                randomized_coords = refined_coords.copy()
                randomized_coords += np.random.normal(0, random_scale, randomized_coords.shape)
                
                predictions.append(randomized_coords)
                
                if len(predictions) >= n_predictions:
                    break
    
    # If we don't have enough predictions from templates, generate de novo structures
    while len(predictions) < n_predictions:
        seed_value = hash(target_id) % 10000 + len(predictions) * 1000
        de_novo_coords = generate_rna_structure(sequence, seed=seed_value)
        
        # Apply stronger constraints to de novo structures (lower confidence)
        refined_de_novo = adaptive_rna_constraints(de_novo_coords, sequence, confidence=0.2)
        
        predictions.append(refined_de_novo)
    
    return predictions[:n_predictions]
# Initialize counters and range settings
if is_submission_mode:
    DRFOLD_START_IDX = 14
    DRFOLD_END_IDX = len(test_sequences) - 1
else:
    DRFOLD_START_IDX = 0
    DRFOLD_END_IDX = 0

drfold_processed = 0
template_processed = 0

train_coords_dict = process_labels_vectorized(train_labels_final)
# Sort test sequences by length to process shorter ones with DRfold2
test_sequences = test_sequences.sort_values(by=['sequence'], key=lambda x: x.str.len())

# List to store all prediction records
all_predictions = []

# Set up time tracking
start_time = time.time()
total_targets = len(test_sequences)

# For each sequence in the test set
for idx, row in test_sequences.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    
    # Progress tracking
    elapsed = time.time() - start_time
    targets_processed = idx
    if targets_processed > 0:
        avg_time_per_target = elapsed / targets_processed
        est_time_remaining = avg_time_per_target * (total_targets - targets_processed)
        time_left = DRFOLD_TIME_LIMIT - (time.time() - start_time_global)
        print(f"Processing target {targets_processed+1}/{total_targets}: {target_id} ({len(sequence)} nt), "
              f"elapsed: {elapsed:.1f}s, est. remaining: {est_time_remaining:.1f}s, time left: {time_left:.1f}s")
    
    # Check if we should use DRfold2 or template-based approach
    use_drfold = (DRFOLD_START_IDX <= idx <= DRFOLD_END_IDX and 
                 (time.time() - start_time_global) < DRFOLD_TIME_LIMIT)
    
    # Generate 5 different structure predictions
    if use_drfold:
        print(f"Using DRfold2 for target {target_id} (index {idx})")
        predictions = predict_rna_structures_drfold2(sequence, target_id)
        
        # If DRfold2 fails, fall back to template approach
        if predictions is None:
            print(f"DRfold2 failed for {target_id}, falling back to template approach")
            predictions = predict_rna_structures(sequence, target_id, train_seqs_final, train_coords_dict)
            template_processed += 1
        else:
            drfold_processed += 1
    else:
        if idx > DRFOLD_END_IDX:
            reason = "index out of DRfold range"
        elif idx < DRFOLD_START_IDX:
            reason = "index before DRfold start range"
        else:
            reason = "time limit reached"
        print(f"Using template approach for {target_id} ({reason})")
        predictions = predict_rna_structures(sequence, target_id, train_seqs_final, train_coords_dict)
        template_processed += 1
    
    # For each residue in the sequence
    for j in range(len(sequence)):
        pred_row = {
            'ID': f"{target_id}_{j+1}",
            'resname': sequence[j],
            'resid': j + 1
        }
        
        # Add coordinates from all 5 predictions
        for i in range(5):
            pred_row[f'x_{i+1}'] = predictions[i][j][0]
            pred_row[f'y_{i+1}'] = predictions[i][j][1]
            pred_row[f'z_{i+1}'] = predictions[i][j][2]
        
        all_predictions.append(pred_row)
    
    # Free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Create DataFrame with predictions
submission_df = pd.DataFrame(all_predictions)

# Ensure the submission file has the correct format
column_order = ['ID', 'resname', 'resid']
for i in range(1, 6):
    for coord in ['x', 'y', 'z']:
        column_order.append(f'{coord}_{i}')
        
submission_df = submission_df[column_order]

# Clean the working directory before saving
print("Cleaning working directory...")
for item in os.listdir("/kaggle/working/"):
    item_path = os.path.join("/kaggle/working/", item)
    if os.path.isfile(item_path) and item != "submission.csv":
        os.remove(item_path)
    elif os.path.isdir(item_path) and item != "predictions" and item != "fasta_files" and item != "DRfold2":
        shutil.rmtree(item_path)

# Save the submission
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print(f"Submission file saved to /kaggle/working/submission.csv")
print(f"Generated predictions for {len(test_sequences)} RNA sequences")
print(f"Used DRfold2 for {drfold_processed} targets and template approach for {template_processed} targets")
print(f"Total runtime: {time.time() - start_time_global:.1f} seconds")
submission_df
# !rm -rf ./*

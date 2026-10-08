"""
connectome_loader.py
--------------------
Loads the FlyWire v783 connectome from the data directory and builds the
sparse weight matrix used by TorchModel.

Data files required (in brain/data/):
  2025_Completeness_783.csv    — neuron table (138,639 rows)
  2025_Connectivity_783.parquet — synapse table (15,091,983 rows)

The loader caches weight_coo.pkl and weight_csr.pkl alongside the data files
so subsequent loads are fast (skips parquet parsing).
"""

import pickle
import pandas as pd
import torch
from pathlib import Path


_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
_ORIG_DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'fly-brain-full' / 'data'

def _resolve_data_dir():
    """Find data directory — prefer local brain/data/, fall back to fly-brain-full/data/."""
    for d in (_DATA_DIR, _ORIG_DATA_DIR):
        comp = d / '2025_Completeness_783.csv'
        conn = d / '2025_Connectivity_783.parquet'
        if comp.exists() and conn.exists():
            return d, comp, conn
    raise FileNotFoundError(
        "Could not find connectome data files. "
        f"Looked in:\n  {_DATA_DIR}\n  {_ORIG_DATA_DIR}\n"
        "Expected: 2025_Completeness_783.csv  and  2025_Connectivity_783.parquet"
    )


def get_hash_tables(comp_path: Path):
    """
    Build FlyWire neuron ID <-> tensor index mappings.

    Returns:
        flyid2i  dict[int, int]  — FlyWire segment ID → tensor row/column
        i2flyid  dict[int, int]  — tensor index → FlyWire segment ID
    """
    df_comp = pd.read_csv(comp_path, index_col=0)
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    i2flyid = {i: j for i, j in enumerate(df_comp.index)}
    return flyid2i, i2flyid


def get_weights(comp_path: Path, conn_path: Path, cache_dir: Path, csr: bool = True):
    """
    Load or build sparse weight matrix from the FlyWire connectivity parquet.

    Caches weight_coo.pkl / weight_csr.pkl in cache_dir so subsequent calls
    skip the parquet parsing step (first call: ~30–60 s; subsequent: <5 s).

    The 'Excitatory x Connectivity' column encodes sign:
      +N  → excitatory (acetylcholine / glutamate)
      -N  → inhibitory (GABA / glycine)

    Returns:
        torch.sparse tensor  shape (N, N)  — sparse weight matrix
    """
    cache_dir = Path(cache_dir)
    coo_path = cache_dir / 'weight_coo.pkl'
    csr_path = cache_dir / 'weight_csr.pkl'

    data_conn = pd.read_parquet(conn_path)
    data_name = pd.read_csv(comp_path)
    num_neurons = data_name.shape[0]

    # --- COO matrix ---
    try:
        with open(coo_path, 'rb') as f:
            weight_coo = pickle.load(f)
        print(f"[connectome] Loaded COO cache from {coo_path}")
    except FileNotFoundError:
        print('[connectome] Building COO weight matrix from parquet (first run, may take ~1 min)...')
        idx = [
            data_conn['Postsynaptic_Index'].to_list(),
            data_conn['Presynaptic_Index'].to_list(),
        ]
        val = data_conn['Excitatory x Connectivity'].to_list()
        weight_coo = torch.sparse_coo_tensor(
            idx, val, (num_neurons, num_neurons)
        ).to(torch.float32)
        with open(coo_path, 'wb') as f:
            pickle.dump(weight_coo, f)
        print(f"[connectome] COO cache saved to {coo_path}")

    if not csr:
        return weight_coo

    # --- CSR matrix (faster matmul) ---
    try:
        with open(csr_path, 'rb') as f:
            weight_csr = pickle.load(f)
        print(f"[connectome] Loaded CSR cache from {csr_path}")
    except FileNotFoundError:
        print('[connectome] Converting COO -> CSR...')
        weight_csr = weight_coo.to_sparse_csr()
        with open(csr_path, 'wb') as f:
            pickle.dump(weight_csr, f)
        print(f"[connectome] CSR cache saved to {csr_path}")

    return weight_csr


def load_annotations(data_dir: Path = None):
    """
    Load FlyWire neuron type annotations.

    Returns:
        pd.DataFrame with columns including 'root_id', 'cell_type', 'super_class', etc.
        Returns None if the annotation file is missing.
    """
    d = Path(data_dir) if data_dir else _DATA_DIR
    tsv = d / 'flywire_annotations.tsv'
    if not tsv.exists():
        tsv = _ORIG_DATA_DIR / 'flywire_annotations.tsv'
    if not tsv.exists():
        return None
    return pd.read_csv(tsv, sep='\t', low_memory=False)


def load_connectome(device: str = 'cpu', csr: bool = True):
    """
    High-level loader: resolve data, build ID tables and weight matrix.

    Returns:
        flyid2i   dict[int, int]
        i2flyid   dict[int, int]
        weights   torch.sparse tensor  (num_neurons × num_neurons)
        num_neurons  int
    """
    data_dir, comp_path, conn_path = _resolve_data_dir()
    print(f"[connectome] Data directory: {data_dir}")

    flyid2i, i2flyid = get_hash_tables(comp_path)
    num_neurons = len(flyid2i)
    print(f"[connectome] Neurons: {num_neurons:,}")

    weights = get_weights(comp_path, conn_path, data_dir, csr=csr)
    weights = weights.to(device=device)

    nnz = weights.values().shape[0] if hasattr(weights, 'values') else int(weights._nnz())
    print(f"[connectome] Synapses (non-zeros): {nnz:,}")

    return flyid2i, i2flyid, weights, num_neurons

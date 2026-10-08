"""
loader.py
---------
Connectome data loading and sparse weight tensor compilation for FlyWire v783.
Loads neuron tables (138,639 neurons) and synaptic edges (15,091,983 connections).
Caches compiled PyTorch sparse COO and CSR tensors for instant startup.
"""

from pathlib import Path
import pickle
from typing import Tuple, Dict, Optional
import pandas as pd
import torch

import os

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_ORIG_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "fly-brain-full" / "data"


def _resolve_data_dir() -> Tuple[Path, Path, Path]:
    """Find data directory - prefer brain/data/, fall back to external connectome caches."""
    candidates = [_DATA_DIR, _ORIG_DATA_DIR]
    if "FLYWIRE_DATA_DIR" in os.environ:
        candidates.insert(0, Path(os.environ["FLYWIRE_DATA_DIR"]))
    
    # Also check sibling directories relative to repository root
    repo_parent = Path(__file__).resolve().parents[3]
    candidates.append(repo_parent / "fly-brain-full" / "data")
    candidates.append(repo_parent / "MuJoCo" / "fly-brain-full" / "data")

    for d in candidates:
        comp = d / "2025_Completeness_783.csv"
        conn = d / "2025_Connectivity_783.parquet"
        if comp.exists() and conn.exists():
            return d, comp, conn
    looked_in = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find FlyWire connectome data files.\n"
        f"Looked in:\n  {looked_in}\n"
        "Expected: 2025_Completeness_783.csv and 2025_Connectivity_783.parquet"
    )



def get_hash_tables(comp_path: Path) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Build FlyWire neuron segment ID <-> contiguous tensor index mappings.

    Returns:
        flyid2i: FlyWire segment ID -> tensor index [0 .. N-1]
        i2flyid: tensor index [0 .. N-1] -> FlyWire segment ID
    """
    df_comp = pd.read_csv(comp_path, index_col=0)
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    i2flyid = {i: j for i, j in enumerate(df_comp.index)}
    return flyid2i, i2flyid


def get_weights(comp_path: Path, conn_path: Path, cache_dir: Path, csr: bool = True) -> torch.Tensor:
    """
    Load or build sparse weight matrix from the FlyWire connectivity parquet.
    Uses cached pickle files (weight_coo.pkl, weight_csr.pkl) if present.

    The 'Excitatory x Connectivity' column encodes sign:
      +N -> excitatory (acetylcholine / glutamate)
      -N -> inhibitory (GABA / glycine)
    """
    cache_dir = Path(cache_dir)
    coo_path = cache_dir / "weight_coo.pkl"
    csr_path = cache_dir / "weight_csr.pkl"

    data_name = pd.read_csv(comp_path)
    num_neurons = data_name.shape[0]

    # --- COO matrix ---
    try:
        with open(coo_path, "rb") as f:
            weight_coo = pickle.load(f)
        print(f"[connectome] Loaded COO cache from {coo_path}")
    except FileNotFoundError:
        print("[connectome] Building COO weight matrix from parquet (first run, may take ~1 min)...")
        data_conn = pd.read_parquet(conn_path)
        idx = [
            data_conn["Postsynaptic_Index"].to_list(),
            data_conn["Presynaptic_Index"].to_list(),
        ]
        val = data_conn["Excitatory x Connectivity"].to_list()
        weight_coo = torch.sparse_coo_tensor(
            idx, val, (num_neurons, num_neurons)
        ).to(torch.float32)
        with open(coo_path, "wb") as f:
            pickle.dump(weight_coo, f)
        print(f"[connectome] COO cache saved to {coo_path}")

    if not csr:
        return weight_coo

    # --- CSR matrix (faster matmul) ---
    try:
        with open(csr_path, "rb") as f:
            weight_csr = pickle.load(f)
        print(f"[connectome] Loaded CSR cache from {csr_path}")
    except FileNotFoundError:
        print("[connectome] Converting COO -> CSR...")
        weight_csr = weight_coo.to_sparse_csr()
        with open(csr_path, "wb") as f:
            pickle.dump(weight_csr, f)
        print(f"[connectome] CSR cache saved to {csr_path}")

    return weight_csr


def load_annotations(data_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load FlyWire neuron type annotations (cell_type, super_class, etc.)."""
    if data_dir is not None:
        tsv = Path(data_dir) / "flywire_annotations.tsv"
        if tsv.exists():
            return pd.read_csv(tsv, sep="\t", low_memory=False)
    try:
        resolved_dir, _, _ = _resolve_data_dir()
        tsv = resolved_dir / "flywire_annotations.tsv"
        if tsv.exists():
            return pd.read_csv(tsv, sep="\t", low_memory=False)
    except FileNotFoundError:
        pass
    return None



def load_connectome(device: str = "cpu", csr: bool = True) -> Tuple[Dict[int, int], Dict[int, int], torch.Tensor, int]:
    """
    High-level loader: resolves data, builds ID tables, and loads weight matrix.

    Returns:
        flyid2i: dict mapping FlyWire ID -> tensor index
        i2flyid: dict mapping tensor index -> FlyWire ID
        weights: torch.sparse tensor (num_neurons x num_neurons)
        num_neurons: total neuron count (138,639)
    """
    data_dir, comp_path, conn_path = _resolve_data_dir()
    print(f"[connectome] Data directory: {data_dir}")

    flyid2i, i2flyid = get_hash_tables(comp_path)
    num_neurons = len(flyid2i)
    print(f"[connectome] Neurons: {num_neurons:,}")

    weights = get_weights(comp_path, conn_path, data_dir, csr=csr)
    weights = weights.to(device=device)

    nnz = weights.values().shape[0] if hasattr(weights, "values") else int(weights._nnz())
    print(f"[connectome] Synapses (non-zeros): {nnz:,}")

    return flyid2i, i2flyid, weights, num_neurons

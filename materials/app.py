from __future__ import annotations

import gc
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import requests
import torch
from ase.atoms import Atoms as AseAtoms
from ase.io import read
from ase.optimize import BFGS
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.openapi.utils import get_openapi
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ouro.utils import get_custom_openapi, ouro_field
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import modal

here = Path(__file__).parent  # the directory of this file

MINUTES = 60  # seconds

# Define container paths
CONTAINER_PATH = Path("/materials")
MODELS_PATH = CONTAINER_PATH / "models"
DATA_PATH = CONTAINER_PATH / "data"
OUTPUT_PATH = CONTAINER_PATH / "output"

# Create volume with a descriptive name
volume = modal.Volume.from_name("materials-data", create_if_missing=True)

# Model checkpoint URL
CHECKPOINT_URL = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-inf-mpa-20250404.ckpt"

# Configure environment variables for CUDA support
ENV = {
    "DEBIAN_FRONTEND": "noninteractive",
    "LD_LIBRARY_PATH": "/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH",
}

image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
    .apt_install(["git", "gcc", "g++"])
    .env(ENV)
    # .run_commands(
    #     # Help Numba find libcudart.so
    #     "ln -s /opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 "
    #     "/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so"
    # )
    .pip_install(
        "requests",
        "fastapi[standard]",
        "pydantic",
        "numpy",
        "ase",
        "pymatgen",
        "matplotlib",
        "ouro-py",
        "orb-models",
    )
    # .pip_install("cuml-cu12==25.2.*", extra_index_url="https://pypi.nvidia.com")
)

app = modal.App(
    name="materials", image=image, secrets=[modal.Secret.from_name("mattergen")]
)


def get_orb_calculator(
    device: str = "cuda",
    weights_path: str = str(
        MODELS_PATH / "checkpoints" / "orb-v3-conservative-inf-mpa.ckpt"
    ),
):
    """Return an instance of the ORB calculator."""
    orbff = pretrained.orb_v3_conservative_inf_mpa(
        device=device, weights_path=weights_path, compile=False
    )
    return ORBCalculator(orbff, device=device)


def cleanup_memory():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@app.function(
    volumes={str(CONTAINER_PATH): volume},
    timeout=20 * MINUTES,
    image=image,
)
def download_model(force_download: bool = False):
    """Download the Orb model for structure relaxation."""
    # Ensure all necessary directories exist
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = MODELS_PATH / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "orb-v3-conservative-inf-mpa.ckpt"

    if not checkpoint_path.exists() or force_download:
        print(f"Downloading model checkpoint to {checkpoint_path}...")
        response = requests.get(CHECKPOINT_URL, stream=True)
        response.raise_for_status()

        with open(checkpoint_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Commit the downloaded file to the volume
        volume.commit()
        print("Checkpoint downloaded successfully")
    else:
        print(f"Using existing checkpoint at {checkpoint_path}")

    # Initialize model to verify the checkpoint
    print("Initializing model to verify checkpoint...")
    model = get_orb_calculator(device="cuda", weights_path=str(checkpoint_path))
    print("Model initialized successfully")


class File(BaseModel):
    url: str
    filename: str
    name: Optional[str] = None
    description: Optional[str] = None
    id: Optional[str] = None
    type: str
    org_id: str
    team_id: str
    visibility: str


class RelaxStructureRequest(BaseModel):
    file: File
    fmax: float = Field(0.03, ge=0.0, le=1.0, description="Force convergence (eV/Å)")
    max_steps: int = Field(400, ge=1, le=1000, description="Maximum optimization steps")


@app.cls(image=image, volumes={str(CONTAINER_PATH): volume}, gpu="T4")
class FastAPIApp:
    web_app = FastAPI(
        title="Materials API",
        summary="Relax crystal structures using ML interatomic potentials",
        description="API for relaxing crystal structures using machine learning interatomic potentials like Orb.",
        version="1.0.0",
    )

    @modal.enter()
    def setup(self):
        print("Starting FastAPI app...")
        # Initialize calculator
        print("Loading model...")
        self.calculator = get_orb_calculator(device="cuda")
        print("Model loaded successfully")
        # Set up routes
        self.setup_routes()
        # Configure OpenAPI
        self.web_app.openapi = get_custom_openapi(self.web_app, get_openapi)

    def setup_routes(self):
        @self.web_app.get("/materials")
        async def welcome(request: Request):
            return {"message": "Welcome to the Materials API!"}

        @self.web_app.post(
            "/materials/structure/relax",
            summary="Relax a crystal structure",
            description="Optimize a crystal structure using machine learning interatomic potentials",
        )
        @ouro_field("x-ouro-input-asset-type", "file")
        @ouro_field("x-ouro-input-file-extension", "cif")
        @ouro_field("x-ouro-output-asset-type", "file")
        async def relax(
            request: RelaxStructureRequest,
            ouro_route_id: Optional[str] = Header(None, alias="ouro-route-id"),
            ouro_route_org_id: Optional[str] = Header(None, alias="ouro-route-org-id"),
            ouro_route_team_id: Optional[str] = Header(
                None, alias="ouro-route-team-id"
            ),
            ouro_action_id: Optional[str] = Header(None, alias="ouro-action-id"),
        ):
            import base64
            import os
            import time

            from ouro import Ouro

            ouro = Ouro(api_key=os.environ["OURO_API_KEY"])

            try:
                # Fetch the input structure
                resp = requests.get(request.file.url, timeout=30)
                resp.raise_for_status()
                atoms = read(
                    BytesIO(resp.content), format=request.file.filename.split(".")[-1]
                )
                if isinstance(atoms, list):  # multi-model CIF
                    atoms = atoms[0]

                atoms.pbc = True  # Keep it periodic

                # Store trajectory frames in memory
                trajectory_frames: List[AseAtoms] = [atoms.copy()]

                # Convert ASE Atoms to Pymatgen Structure for symmetry analysis
                adaptor = AseAtomsAdaptor()
                # Create a new ASE atoms object that pymatgen can handle
                ase_atoms = AseAtoms(
                    symbols=atoms.get_chemical_symbols(),
                    positions=atoms.get_positions(),
                    cell=atoms.get_cell(),
                    pbc=atoms.get_pbc(),
                )
                # First convert to Structure
                input_structure: Structure = adaptor.get_structure(ase_atoms)
                input_spg = SpacegroupAnalyzer(input_structure, symprec=0.1)
                input_symmetry = {
                    "space_group_number": input_spg.get_space_group_number(),
                    "space_group_symbol": input_spg.get_space_group_symbol(),
                    "crystal_system": input_spg.get_crystal_system(),
                    "point_group": input_spg.get_point_group_symbol(),
                    "is_centrosymmetric": input_spg.is_laue(),
                }

                ouro.client.post(
                    f"/actions/{ouro_action_id}/log",
                    json={
                        "message": f"Starting relaxation. {len(atoms)} atoms, {input_symmetry['space_group_symbol']} symmetry.",
                        "asset_id": ouro_route_id,
                        "level": "info",
                    },
                )

                print("Loading model...")
                start_time = time.time()

                # Use cached calculator
                atoms.calc = self.calculator
                starting_energy = atoms.get_potential_energy()

                end_time = time.time()
                print(f"Model loaded in {end_time - start_time:.2f} seconds")

                ouro.client.post(
                    f"/actions/{ouro_action_id}/log",
                    json={
                        "message": f"Model loaded in {end_time - start_time:.2f} seconds",
                        "asset_id": ouro_route_id,
                        "level": "info",
                    },
                )

                # Optimize
                dyn = BFGS(atoms)
                dyn.attach(lambda: trajectory_frames.append(atoms.copy()))

                def log_step():
                    if dyn.nsteps == 0 or dyn.nsteps % 10 == 0:
                        current_energy = atoms.get_potential_energy()
                        energy_change = current_energy - starting_energy
                        ouro.client.post(
                            f"/actions/{ouro_action_id}/log",
                            json={
                                "message": f"Step {dyn.nsteps}, {current_energy:.4f} eV (Δ{energy_change:+.4f} eV)",
                                "asset_id": ouro_route_id,
                                "level": "info",
                            },
                        )

                dyn.attach(log_step)

                try:
                    dyn.run(fmax=request.fmax, steps=request.max_steps)

                    # Log final step
                    final_energy = atoms.get_potential_energy()
                    energy_change = final_energy - starting_energy
                    ouro.client.post(
                        f"/actions/{ouro_action_id}/log",
                        json={
                            "message": f"Finished in {dyn.nsteps} steps. Final energy: {final_energy:.4f} eV (Δ{energy_change:+.4f} eV)",
                            "asset_id": ouro_route_id,
                            "level": "info",
                        },
                    )
                except RuntimeError as exc:
                    raise HTTPException(
                        500,
                        detail=f"Relaxation failed after {dyn.nsteps} steps; "
                        f"max |F| ≈ {atoms.get_forces().max():.3f} eV/Å",
                    ) from exc

                # Convert ASE Atoms to Pymatgen Structure for CIF writing
                # Create a new ASE atoms object that pymatgen can handle
                ase_atoms = AseAtoms(
                    symbols=atoms.get_chemical_symbols(),
                    positions=atoms.get_positions(),
                    cell=atoms.get_cell(),
                    pbc=atoms.get_pbc(),
                )
                # First convert to Structure
                structure: Structure = adaptor.get_structure(ase_atoms)

                # Analyze output symmetry
                output_spg = SpacegroupAnalyzer(structure, symprec=0.1)
                output_symmetry = {
                    "space_group_number": output_spg.get_space_group_number(),
                    "space_group_symbol": output_spg.get_space_group_symbol(),
                    "crystal_system": output_spg.get_crystal_system(),
                    "point_group": output_spg.get_point_group_symbol(),
                    "is_centrosymmetric": output_spg.is_laue(),
                }

                # Write CIF with symmetry information, using conventional cell to preserve supercell
                # Determine if this is a supercell by comparing with primitive cell
                primitive_structure = output_spg.find_primitive()
                is_supercell = len(structure) > len(primitive_structure)

                # Write CIF with symmetry information
                writer = CifWriter(
                    structure,
                    symprec=0.1,
                    write_magmoms=False,
                    significant_figures=8,
                    write_site_properties=False,
                    refine_struct=not is_supercell,  # Only refine if it's a unit cell
                )
                cif_string = str(writer)
                cif64 = base64.b64encode(cif_string.encode("utf-8")).decode("utf-8")

                desc = (
                    f"Relaxed with Orb v3; "
                    f"{request.fmax} eV/Å threshold; "
                    f"final energy = {final_energy:.4f} eV; "
                    f"energy change = {energy_change:.4f} eV; "
                    f"symmetry: {input_symmetry['space_group_symbol']} → {output_symmetry['space_group_symbol']}"
                )

                # Clean up memory after processing
                cleanup_memory()

                return {
                    "file": {
                        "name": (request.file.name or atoms.get_chemical_formula())
                        + " - relaxed",
                        "description": desc,
                        "filename": request.file.filename.rsplit(".", 1)[0]
                        + "_relaxed.cif",
                        "type": "text/cif",  # important to make sure download mimetype is right
                        "extension": "cif",
                        "base64": cif64,
                        "org_id": request.file.org_id,
                        "team_id": request.file.team_id,
                        "visibility": request.file.visibility,
                    },
                    "starting_energy": starting_energy,
                    "optimized_energy": final_energy,
                    "steps": dyn.nsteps,
                    "input_symmetry": input_symmetry,
                    "output_symmetry": output_symmetry,
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    @modal.asgi_app()
    def serve(self):
        return self.web_app

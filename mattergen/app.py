from pathlib import Path
from typing import Any, Dict, Optional, Union
from uuid import uuid4

import requests
from fastapi import FastAPI, Header
from fastapi.openapi.utils import get_openapi
from ouro.utils import get_custom_openapi, ouro_field
from pydantic import BaseModel, Field

import modal

here = Path(__file__).parent  # the directory of this file

MINUTES = 60  # seconds

# Define container paths
CONTAINER_PATH = Path("/mattergen")
MODELS_PATH = CONTAINER_PATH / "models"
DATA_PATH = CONTAINER_PATH / "data"
OUTPUT_PATH = CONTAINER_PATH / "output"

# Create volume with a descriptive name
volume = modal.Volume.from_name("mattergen-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "git-lfs", "python3-distutils", "python3-setuptools"])
    # First install PyTorch and related packages from PyTorch index
    .pip_install(
        "torch==2.2.1+cu118",
        "torchvision==0.17.1+cu118",
        "torchaudio==2.2.1+cu118",
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
    # Then install PyG packages from their specific wheels
    .pip_install(
        "https://data.pyg.org/whl/torch-2.2.0+cu118/pyg_lib-0.4.0+pt22cu118-cp310-cp310-linux_x86_64.whl",
        "https://data.pyg.org/whl/torch-2.2.0+cu118/torch_cluster-1.6.3+pt22cu118-cp310-cp310-linux_x86_64.whl",
        "https://data.pyg.org/whl/torch-2.2.0+cu118/torch_scatter-2.1.2+pt22cu118-cp310-cp310-linux_x86_64.whl",
        "https://data.pyg.org/whl/torch-2.2.0+cu118/torch_sparse-0.6.18+pt22cu118-cp310-cp310-linux_x86_64.whl",
    )
    # Then install other requirements
    .pip_install(
        "requests",
        "fastapi[standard]",
        "pydantic",
        "numpy<2.0",
        "scikit-learn",
        "tqdm==4.65.0",
        "pyzmq==25.1.1",
        "omegaconf==2.3.0",
        "pyyaml==6.0.1",
        "ouro-py",
        "ase>=3.22.1",
        "pymatgen>=2024.6.4",
        "mattersim>=1.1",
        "autopep8",
        "cachetools",
        "contextlib2",
        "emmet-core>=0.84.2",  # keep up-to-date together with pymatgen, atomate2
        "fire",  # see https://github.com/google/python-fire
        "huggingface-hub",
        "hydra-core==1.3.1",
        "hydra-joblib-launcher==1.1.5",
        "jupyterlab>=4.2.5",
        "lmdb",
        "matplotlib==3.8.4",
        "matscipy>=0.7.0",
        "mattersim>=1.1",
        "monty==2024.7.30 ",  # keep up-to-date together with pymatgen, atomate2
        "notebook>=7.2.2",
        "numpy<2.0",  # pin numpy before breaking changes in 2.0
        "omegaconf==2.3.0",
        "pymatgen>=2024.6.4",
        "pylint",
        "pytest",
        "pytorch-lightning==2.0.6",
        "seaborn>=0.13.2",  # for plotting
        "setuptools",
        "SMACT",
        "sympy>=1.11.1",
        "torch==2.2.1+cu118; sys_platform == 'linux'",
        "torchvision==0.17.1+cu118; sys_platform == 'linux'",
        "torchaudio==2.2.1+cu118; sys_platform == 'linux'",
        "torch==2.4.1; sys_platform == 'darwin'",
        "torchvision==0.19.1; sys_platform == 'darwin'",
        "torchaudio==2.4.1; sys_platform == 'darwin'",
        "torch_cluster",
        "torch_geometric>=2.5",
        "torch_scatter",
        "torch_sparse",
        "tqdm",
        "wandb>=0.10.33",
    )
    .run_commands(
        "git lfs install",
        "git clone https://github.com/microsoft/mattergen.git /opt/mattergen",
        # Install mattergen in development mode but skip dependencies since we installed them above
        "cd /opt/mattergen && pip install -e . --no-deps",
    )
)

app = modal.App(
    name="mattergen", image=image, secrets=[modal.Secret.from_name("mattergen")]
)


class BaseGenerationRequest(BaseModel):
    """Base class for all generation requests."""

    batch_size: int = Field(
        16, ge=1, le=64, description="Number of structures to generate in each batch"
    )
    guidance_factor: float = Field(
        2.0,
        ge=0.0,
        le=10.0,
        description="Diffusion guidance factor. Higher values produce samples that better match the target property but may be less realistic.",
    )


class ChemicalSystemRequest(BaseGenerationRequest):
    """Request model for chemical system generation."""

    chemical_system: str = Field(
        ..., description="Chemical system to generate (e.g., 'Li-O' or 'Fe-Co-Ni')"
    )


class MagneticDensityRequest(BaseGenerationRequest):
    """Request model for magnetic density generation."""

    magnetic_density: float = Field(
        ...,
        ge=0.000,
        le=0.300,
        description="Target magnetic density value in units of Angstrom^-3",
    )


async def run_mattergen(
    output_dir: Path,
    model_name: str,
    batch_size: int,
    properties_to_condition_on: Dict[str, Any],
    guidance_factor: float,
    ouro_client: Any = None,
    ouro_action_id: Optional[str] = None,
    ouro_route_id: Optional[str] = None,
) -> tuple[int, str, str]:
    """Run MatterGen model and process results.

    Args:
        output_dir: Directory to save generated structures
        model_name: Name of the pre-trained model to use
        batch_size: Number of structures to generate per batch
        properties_to_condition_on: Dict of properties to condition on
        guidance_factor: Diffusion guidance factor
        ouro_client: Optional Ouro client for logging
        ouro_action_id: Optional Ouro action ID for logging
        ouro_route_id: Optional Ouro route ID for logging

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    import re
    import subprocess

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    if ouro_client and ouro_action_id:
        ouro_client.post(
            f"/actions/{ouro_action_id}/log",
            json={
                "message": "Starting generation...",
                "asset_id": ouro_route_id,
                "level": "info",
            },
        )

    # Build the command
    command = [
        "mattergen-generate",
        str(output_dir),
        f"--pretrained-name={model_name}",
        f"--batch_size={batch_size}",
        f"--num_batches=1",
        f"--properties_to_condition_on={properties_to_condition_on}",
        f"--diffusion_guidance_factor={guidance_factor}",
        "--record_trajectories=False",
    ]

    print("Running command:", command, sep="\n\t")

    progress_pattern = re.compile(r"(\d+)%\|")
    last_progress = None

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    stdout_lines = []
    stderr_lines = []

    # Stream output in real-time
    while True:
        stdout_line = process.stdout.readline() if process.stdout else ""
        stderr_line = process.stderr.readline() if process.stderr else ""

        if stderr_line:
            stderr_lines.append(stderr_line)
            # Check if this is a progress line
            progress_match = progress_pattern.search(stderr_line)
            if progress_match and ouro_client and ouro_action_id:
                current_progress = int(progress_match.group(1))
                # Only log if progress has changed and is divisible by 5
                if current_progress != last_progress and current_progress % 5 == 0:
                    ouro_client.post(
                        f"/actions/{ouro_action_id}/log",
                        json={
                            "message": f"Generation progress: {current_progress}%",
                            "asset_id": ouro_route_id,
                            "level": "info",
                        },
                    )
                    last_progress = current_progress

        if process.poll() is not None:
            break

    # Get the return code
    returncode = process.wait()
    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)

    return returncode, stdout, stderr


@app.function(
    volumes={str(CONTAINER_PATH): volume},  # Mount at the root container path
    timeout=20 * MINUTES,
    image=image,
)
def download_model(
    force_download: bool = False,
    model_name: str = "chemical_system",
):
    """Download the MatterGen model checkpoint."""
    import subprocess

    # Ensure all necessary directories exist
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # The model should be downloaded from Hugging Face
    # This happens automatically when we try to use it
    # But we should verify the model exists
    model_path = MODELS_PATH / model_name

    if not force_download:
        if model_path.exists():
            print(f"Model already downloaded to {MODELS_PATH}")
            return

    print(f"Model will be downloaded automatically when first used")
    volume.commit()


def check_symmetry(structures_path: str) -> bool:
    """Check and fix symmetry of generated structures.

    Args:
        structures_path: Path to the directory containing generated structures

    Returns:
        bool: True if successful, False otherwise
    """
    import os
    import tempfile
    from pathlib import Path
    from zipfile import ZipFile

    import ase.io
    from pymatgen.core import Element, Lattice, Structure
    from pymatgen.io.cif import CifWriter
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        print(f"Starting symmetry correction for {structures_path}")

        # Load the structures from the EXTXYZ file that mattergen generates
        extxyz_file = Path(OUTPUT_PATH) / structures_path / "generated_crystals.extxyz"
        print(f"Looking for EXTXYZ file at {extxyz_file}")

        if not extxyz_file.exists():
            print(f"Could not find EXTXYZ file at {extxyz_file}")
            return False

        # Read structures from EXTXYZ file
        print("Reading structures from EXTXYZ file...")
        atoms_list = ase.io.read(str(extxyz_file), index=":")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        print(f"Found {len(atoms_list)} structures to process")

        # Create output directory if it doesn't exist
        output_dir = Path(OUTPUT_PATH) / structures_path
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created/verified output directory at {output_dir}")

        successful_structures = 0
        failed_structures = 0

        # Create a temporary directory for the new CIF files
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Created temporary directory at {temp_dir}")

            # Process each structure
            for idx, atoms in enumerate(atoms_list):
                try:
                    print(f"\nProcessing structure {idx + 1}/{len(atoms_list)}")
                    print(f"Structure {idx} has {len(atoms)} atoms")
                    print(f"Cell parameters: {atoms.cell}")

                    # Convert ASE atoms to pymatgen Structure manually
                    structure = Structure(
                        lattice=Lattice(atoms.cell),
                        species=[
                            Element(str(atoms.symbols[i])) for i in range(len(atoms))
                        ],
                        coords=atoms.positions,
                        coords_are_cartesian=True,
                    )
                    print(f"Successfully converted to pymatgen Structure")

                    # Analyze symmetry
                    print("Analyzing symmetry...")
                    sga = SpacegroupAnalyzer(structure, symprec=0.1)
                    sym_structure = sga.get_refined_structure()
                    spacegroup = sga.get_space_group_symbol()
                    print(f"Found space group: {spacegroup}")

                    # Write CIF with proper symmetry
                    temp_cif = os.path.join(temp_dir, f"gen_{idx}.cif")
                    writer = CifWriter(
                        sym_structure,
                        symprec=1e-3,
                        write_magmoms=False,
                        significant_figures=8,
                    )
                    writer.write_file(temp_cif)
                    print(f"Successfully wrote CIF file to {temp_cif}")
                    successful_structures += 1

                except Exception as e:
                    print(f"Error processing structure {idx}: {str(e)}")
                    failed_structures += 1
                    # Skip this structure but continue processing others
                    continue

            print(f"\nProcessed {len(atoms_list)} structures:")
            print(f"Successful: {successful_structures}")
            print(f"Failed: {failed_structures}")

            # Create new zip file with corrected structures
            output_zip = output_dir / "generated_crystals_cif_fixed.zip"
            try:
                print(f"Creating zip file at {output_zip}")
                with ZipFile(output_zip, "w") as zip_obj:
                    files_added = 0
                    for idx in range(len(atoms_list)):
                        temp_cif = os.path.join(temp_dir, f"gen_{idx}.cif")
                        if os.path.exists(
                            temp_cif
                        ):  # Only add if file was successfully created
                            zip_obj.write(temp_cif, arcname=f"gen_{idx}.cif")
                            files_added += 1
                    print(f"Added {files_added} files to zip")

                # Verify the zip file was created and contains files
                if output_zip.exists() and output_zip.stat().st_size > 0:
                    print("Successfully created zip file with corrected structures")
                    return True
                else:
                    print("No structures were successfully processed")
                    return False

            except Exception as e:
                print(f"Error creating zip file: {str(e)}")
                return False

    except Exception as e:
        print(f"Error in check_symmetry: {str(e)}")
        return False


@app.function(
    image=image, volumes={str(CONTAINER_PATH): volume}, gpu="T4", timeout=20 * MINUTES
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Request

    from ouro import Ouro

    web_app = FastAPI(
        title="MatterGen",
        summary="Generate crystal structures with MatterGen",
        description="MatterGen is a generative model for inorganic materials design that can be fine-tuned to steer the generation towards a wide range of property constraints.",
        version="1.0.0",
    )

    web_app.openapi = get_custom_openapi(web_app, get_openapi)

    @web_app.get("/mattergen")
    async def welcome(request: Request):
        return {
            "message": "Welcome to MatterGen! Use the /generate endpoints to create new materials."
        }

    async def process_generation_result(
        returncode: int,
        stdout: str,
        stderr: str,
        output_dir: Path,
        description: str,
        filename: str,
        ouro_client: Any,
        ouro_action_id: Optional[str],
        ouro_route_id: Optional[str],
        ouro_route_org_id: Optional[str],
        ouro_route_team_id: Optional[str],
    ):
        """Process generation results and create response."""
        import base64

        if returncode != 0:
            if ouro_action_id:
                ouro_client.post(
                    f"/actions/{ouro_action_id}/log",
                    json={
                        "message": "Generation failed",
                        "asset_id": ouro_route_id,
                        "level": "error",
                    },
                )
            # Print stdout and stderr
            print("stdout", stdout)
            print("stderr", stderr)

            return {
                "error": "Generation failed",
                "stdout": stdout,
                "stderr": stderr,
            }

        # Find the generated CIF files
        cif_zip = output_dir / "generated_crystals_cif.zip"
        if not cif_zip.exists():
            return {
                "error": "No CIF files were generated",
                "stdout": stdout,
                "stderr": stderr,
            }

        # Run symmetry correction directly in this process
        success = check_symmetry(str(output_dir))

        # Read the symmetry-corrected zip file if it exists
        cif_zip_fixed = output_dir / "generated_crystals_cif_fixed.zip"

        if success and cif_zip_fixed.exists():
            try:
                with open(cif_zip_fixed, "rb") as f:
                    zip_bytes_fixed = f.read()
                zip64_fixed = base64.b64encode(zip_bytes_fixed).decode("utf-8")

                if ouro_action_id:
                    ouro_client.post(
                        f"/actions/{ouro_action_id}/log",
                        json={
                            "message": "Successfully generated structures with corrected symmetry",
                            "asset_id": ouro_route_id,
                            "level": "info",
                        },
                    )

                return {
                    "file": {
                        "name": f"{description} (corrected symmetry)",
                        "description": f"{description} - with corrected symmetry",
                        "filename": filename.replace(".zip", "_fixed.zip"),
                        "type": "application/zip",
                        "extension": "zip",
                        "base64": zip64_fixed,
                        "org_id": ouro_route_org_id,
                        "team_id": ouro_route_team_id,
                    }
                }
            except Exception as e:
                print(f"Error reading symmetry-corrected file: {str(e)}")
                # Fall through to return original files

        # If symmetry correction failed or we couldn't read the corrected file,
        # return the original CIF files
        print(
            "Symmetry correction failed or file not readable, returning original CIF files"
        )
        try:
            with open(cif_zip, "rb") as f:
                zip_bytes = f.read()
            zip64 = base64.b64encode(zip_bytes).decode("utf-8")

            if ouro_action_id:
                ouro_client.post(
                    f"/actions/{ouro_action_id}/log",
                    json={
                        "message": "Generated structures (symmetry correction failed)",
                        "asset_id": ouro_route_id,
                        "level": "warning",
                    },
                )

            return {
                "file": {
                    "name": description,
                    "description": f"{description} (P1 symmetry - symmetry correction failed)",
                    "filename": filename,
                    "type": "application/zip",
                    "extension": "zip",
                    "base64": zip64,
                    "org_id": ouro_route_org_id,
                    "team_id": ouro_route_team_id,
                }
            }
        except Exception as e:
            print(f"Error reading original CIF file: {str(e)}")
            return {
                "error": "Failed to read generated structures",
                "stdout": stdout,
                "stderr": stderr,
            }

    # @web_app.post(
    #     "/mattergen/generate/chemical-system",
    #     summary="Generate crystal structures with chemical system conditioning",
    #     description="Generate crystal structures conditioned on a chemical system using MatterGen",
    # )
    @web_app.post(
        "/mattergen/generate",
        summary="Generate crystal structures with chemical system conditioning",
        description="Generate crystal structures conditioned on a chemical system using MatterGen",
    )
    @ouro_field("x-ouro-output-asset-type", "file")
    async def generate_chemical_system(
        request: ChemicalSystemRequest,
        ouro_route_id: Optional[str] = Header(None, alias="ouro-route-id"),
        ouro_route_org_id: Optional[str] = Header(None, alias="ouro-route-org-id"),
        ouro_route_team_id: Optional[str] = Header(None, alias="ouro-route-team-id"),
        ouro_action_id: Optional[str] = Header(None, alias="ouro-action-id"),
    ):
        import os

        ouro = Ouro(api_key=os.environ["OURO_API_KEY"])

        generation_id = uuid4()
        output_dir = OUTPUT_PATH / str(generation_id)

        # Run generation
        returncode, stdout, stderr = await run_mattergen(
            output_dir=output_dir,
            model_name="chemical_system",
            batch_size=request.batch_size,
            properties_to_condition_on={"chemical_system": request.chemical_system},
            guidance_factor=request.guidance_factor,
            ouro_client=ouro.client,
            ouro_action_id=ouro_action_id,
            ouro_route_id=ouro_route_id,
        )

        return await process_generation_result(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            output_dir=output_dir,
            description=f"Generated crystal structures for {request.chemical_system}",
            filename=f"{request.chemical_system}.zip",
            ouro_client=ouro.client,
            ouro_action_id=ouro_action_id,
            ouro_route_id=ouro_route_id,
            ouro_route_org_id=ouro_route_org_id,
            ouro_route_team_id=ouro_route_team_id,
        )

    @web_app.post(
        "/mattergen/generate/magnetic-density",
        summary="Generate crystal structures with magnetic density conditioning",
        description="Generate crystal structures conditioned on magnetic density using MatterGen",
    )
    @ouro_field("x-ouro-output-asset-type", "file")
    async def generate_magnetic_density(
        request: MagneticDensityRequest,
        ouro_route_id: Optional[str] = Header(None, alias="ouro-route-id"),
        ouro_route_org_id: Optional[str] = Header(None, alias="ouro-route-org-id"),
        ouro_route_team_id: Optional[str] = Header(None, alias="ouro-route-team-id"),
        ouro_action_id: Optional[str] = Header(None, alias="ouro-action-id"),
    ):
        import os

        ouro = Ouro(api_key=os.environ["OURO_API_KEY"])

        generation_id = uuid4()
        output_dir = OUTPUT_PATH / str(generation_id)

        # Run generation
        returncode, stdout, stderr = await run_mattergen(
            output_dir=output_dir,
            model_name="dft_mag_density",
            batch_size=request.batch_size,
            properties_to_condition_on={"dft_mag_density": request.magnetic_density},
            guidance_factor=request.guidance_factor,
            ouro_client=ouro.client,
            ouro_action_id=ouro_action_id,
            ouro_route_id=ouro_route_id,
        )

        return await process_generation_result(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            output_dir=output_dir,
            description=f"Generated crystal structures with magnetic density {request.magnetic_density}",
            filename=f"magnetic_density_{request.magnetic_density}.zip",
            ouro_client=ouro.client,
            ouro_action_id=ouro_action_id,
            ouro_route_id=ouro_route_id,
            ouro_route_org_id=ouro_route_org_id,
            ouro_route_team_id=ouro_route_team_id,
        )

    return web_app

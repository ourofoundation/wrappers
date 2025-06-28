from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

import requests
from fastapi import FastAPI, Header
from fastapi.openapi.utils import get_openapi
from ouro.utils import get_custom_openapi, ouro_field
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import modal

here = Path(__file__).parent  # the directory of this file

MINUTES = 60  # seconds

# Define container paths
CONTAINER_PATH = Path("/crystallm")
MODELS_PATH = CONTAINER_PATH / "models"
DATA_PATH = CONTAINER_PATH / "data"
OUTPUT_PATH = CONTAINER_PATH / "output"

# Create volume with a descriptive name
volume = modal.Volume.from_name("crystallm-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "python3-distutils", "python3-setuptools"])
    .pip_install(
        "requests",
        "torch==2.7.0",
        "fastapi[standard]",
        "pydantic",
        "pandas",
        "pymatgen",
        "numpy==1.24.2",
        "scikit-learn",
        "tqdm==4.65.0",
        "pyzmq==25.1.1",
        "omegaconf==2.3.0",
        "pyyaml==6.0.1",
        "smact==2.5.5",
        "matminer==0.9.0",
        "ouro-py",
    )
    .run_commands(
        "git clone https://github.com/lantunes/CrystaLLM.git /opt/crystallm",
        "cd /opt/crystallm && pip install -e .",
    )
)
app = modal.App(
    name="crystallm", image=image, secrets=[modal.Secret.from_name("crystallm")]
)


class SpaceGroup(str, Enum):
    # Triclinic
    P1 = "P1"
    P1_BAR = "P-1"

    # Monoclinic
    P2 = "P2"
    P21 = "P2_1"
    C2 = "C2"
    PM = "Pm"
    PC = "Pc"
    CM = "Cm"
    CC = "Cc"
    P2_M = "P2/m"
    P21_M = "P2_1/m"
    C2_M = "C2/m"
    P2_C = "P2/c"
    P21_C = "P2_1/c"
    C2_C = "C2/c"

    # Orthorhombic
    P222 = "P222"
    P2221 = "P222_1"
    P21212 = "P2_12_12"
    P212121 = "P2_12_12_1"
    C222 = "C222"
    C2221 = "C222_1"
    F222 = "F222"
    I222 = "I222"
    I212121 = "I2_12_12_1"
    PMM2 = "Pmm2"
    PCC2 = "Pcc2"
    PMA2 = "Pma2"
    PCA21 = "Pca2_1"
    PNC2 = "Pnc2"
    PMN21 = "Pmn2_1"
    PBA2 = "Pba2"
    PNA21 = "Pna2_1"
    CMM2 = "Cmm2"
    CCC2 = "Ccc2"
    AMM2 = "Amm2"
    ABM2 = "Abm2"
    AMA2 = "Ama2"
    ABA2 = "Aba2"
    PMMM = "Pmmm"
    PNNN = "Pnnn"
    PCCM = "Pccm"
    PBAN = "Pban"
    PMMA = "Pmma"
    PNNA = "Pnna"
    PMNA = "Pmna"
    PCCA = "Pcca"
    PBAM = "Pbam"
    PCCN = "Pccn"
    PBCM = "Pbcm"
    PNNM = "Pnnm"
    PMMN = "Pmmn"
    PBCN = "Pbcn"
    PBCA = "Pbca"
    PNMA = "Pnma"
    CMCM = "Cmcm"
    CMCA = "Cmca"
    CMMM = "Cmmm"
    CCCM = "Cccm"
    CMMA = "Cmma"
    CCCA = "Ccca"
    FMMM = "Fmmm"
    FDDD = "Fddd"
    IMMM = "Immm"
    IBAM = "Ibam"
    IBCA = "Ibca"
    IMMA = "Imma"

    # Tetragonal
    P4 = "P4"
    P41 = "P4_1"
    P42 = "P4_2"
    P43 = "P4_3"
    I4 = "I4"
    I41 = "I4_1"
    P4_M = "P4/m"
    P42_M = "P4_2/m"
    P4_N = "P4/n"
    P42_N = "P4_2/n"
    I4_M = "I4/m"
    I41_A = "I4_1/a"
    P422 = "P422"
    P4212 = "P42_12"
    P4122 = "P4_122"
    P41212 = "P4_12_12"
    P4222 = "P4_222"
    P42212 = "P4_22_12"
    P4322 = "P4_322"
    P43212 = "P4_32_12"
    I422 = "I422"
    I4122 = "I4_122"
    P4MM = "P4mm"
    P4BM = "P4bm"
    P42CM = "P4_2cm"
    P42NM = "P4_2nm"
    P4CC = "P4cc"
    P4NC = "P4nc"
    P42MC = "P4_2mc"
    P42BC = "P4_2bc"
    I4MM = "I4mm"
    I4CM = "I4cm"
    I41MD = "I4_1md"
    I41CD = "I4_1cd"
    P4_MMM = "P4/mmm"
    P4_MCC = "P4/mcc"
    P4_NBM = "P4/nbm"
    P4_NNC = "P4/nnc"
    P4_MBM = "P4/mbm"
    P4_MNC = "P4/mnc"
    P4_NMM = "P4/nmm"
    P4_NCC = "P4/ncc"
    P42_MMC = "P4_2/mmc"
    P42_MCM = "P4_2/mcm"
    P42_NBC = "P4_2/nbc"
    P42_NNM = "P4_2/nnm"
    P42_MBC = "P4_2/mbc"
    P42_MNM = "P4_2/mnm"
    P42_NMC = "P4_2/nmc"
    P42_NCM = "P4_2/ncm"
    I4_MMM = "I4/mmm"
    I4_MCM = "I4/mcm"
    I41_AMD = "I4_1/amd"
    I41_ACD = "I4_1/acd"

    # Trigonal
    P3 = "P3"
    P31 = "P3_1"
    P32 = "P3_2"
    R3 = "R3"
    P3_BAR = "P-3"
    R3_BAR = "R-3"
    P312 = "P312"
    P321 = "P321"
    P3112 = "P3_112"
    P3121 = "P3_121"
    P3212 = "P3_212"
    P3221 = "P3_221"
    R32 = "R32"
    P3M1 = "P3m1"
    P31M = "P31m"
    P3C1 = "P3c1"
    P31C = "P31c"
    R3M = "R3m"
    R3C = "R3c"
    P3_BAR_1M = "P-31m"
    P3_BAR_M1 = "P-3m1"
    P3_BAR_1C = "P-31c"
    P3_BAR_C1 = "P-3c1"
    R3_BAR_M = "R-3m"
    R3_BAR_C = "R-3c"

    # Hexagonal
    P6 = "P6"
    P61 = "P6_1"
    P65 = "P6_5"
    P62 = "P6_2"
    P64 = "P6_4"
    P63 = "P6_3"
    P6_M = "P6/m"
    P63_M = "P6_3/m"
    P622 = "P622"
    P6122 = "P6_122"
    P6522 = "P6_522"
    P6222 = "P6_222"
    P6422 = "P6_422"
    P6322 = "P6_322"
    P6MM = "P6mm"
    P6CC = "P6cc"
    P63CM = "P6_3cm"
    P63MC = "P6_3mc"
    P6_MMM = "P6/mmm"
    P6_MCC = "P6/mcc"
    P63_MCM = "P6_3/mcm"
    P63_MMC = "P6_3/mmc"

    # Cubic
    P23 = "P23"
    F23 = "F23"
    I23 = "I23"
    P213 = "P2_13"
    I213 = "I2_13"
    PM3 = "Pm3"
    PN3 = "Pn3"
    FM3 = "Fm3"
    FD3 = "Fd3"
    IM3 = "Im3"
    PA3 = "Pa3"
    IA3 = "Ia3"
    P432 = "P432"
    P4232 = "P4_232"
    F432 = "F432"
    F4132 = "F4_132"
    I432 = "I432"
    P4332 = "P4_332"
    P4132 = "P4_132"
    I4132 = "I4_132"
    P43M = "P4_3m"
    F43M = "F4_3m"
    I43M = "I4_3m"
    P43N = "P4_3n"
    F43C = "F4_3c"
    I43D = "I4_3d"
    PM3M = "Pm3m"
    PN3N = "Pn3n"
    PM3N = "Pm3n"
    PN3M = "Pn3m"
    FM3M = "Fm3m"
    FM3C = "Fm3c"
    FD3M = "Fd3m"
    FD3C = "Fd3c"
    IM3M = "Im3m"
    IA3D = "Ia3d"


class GenerationRequest(BaseModel):
    composition: str = Field(
        ..., description="Chemical composition (e.g., 'PbTe' or 'Bi2Se3')"
    )
    space_group: Optional[SpaceGroup] = Field(
        None, description="Space group symbol (e.g., 'Fd3m', 'P4_2/n', etc.)"
    )
    max_new_tokens: int = Field(
        3000, ge=1, le=10000, description="Number of tokens generated in each sample"
    )
    temperature: float = Field(
        0.8,
        ge=0.1,
        le=10.0,
        description="Sampling temperature. 1.0 = no change, < 1.0 = less random, > 1.0 = more random",
    )


@app.function(
    volumes={str(CONTAINER_PATH): volume},  # Mount at the root container path
    timeout=20 * MINUTES,
    image=image,
)
def download_model(
    force_download: bool = False,
    model_name: str = "crystallm_v1_small",
):
    import subprocess

    # Ensure all necessary directories exist
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    BLOCK_SIZE = 1024
    file_name = f"{model_name}.tar.gz"
    url = f"https://zenodo.org/records/10642388/files/{file_name}"
    out_path = MODELS_PATH / model_name
    out_file_path = out_path / file_name

    # Create the specific model directory
    out_path.mkdir(parents=True, exist_ok=True)

    if not force_download:
        if out_path.exists():
            print(f"Model already downloaded to {MODELS_PATH}")
            return

    print(f"Downloading to {out_file_path} ...")

    response = requests.get(url, stream=True)

    with open(out_file_path, "wb") as f:
        for data in response.iter_content(BLOCK_SIZE):
            f.write(data)

    # Extract tar.gz file
    command = ["tar", "-xzf", out_file_path, "-C", out_path, "--strip-components=1"]
    subprocess.run(command, check=True)

    volume.commit()

    print(f"Model downloaded to {out_path}")


@app.function(image=image, volumes={str(CONTAINER_PATH): volume}, gpu="T4")
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Request

    web_app = FastAPI(
        title="CrystaLLM",
        summary="Generate crystal structures with CrystaLLM",
        description="CrystaLLM is a model that generates crystal structures from chemical compositions.",
        version="1.0.0",
    )

    web_app.openapi = get_custom_openapi(web_app, get_openapi)

    @web_app.get("/")
    async def welcome(request: Request):
        return {"message": "Hello, World! We will put additional information here."}

    @web_app.post(
        "/generate",
        summary="Generate a crystal structure with CrystaLLM",
        description="Generate a crystal structure with CrystaLLM",
    )
    @ouro_field("x-ouro-output-asset-type", "file")
    async def generate(
        request: GenerationRequest,
        ouro_route_id: Optional[str] = Header(None, alias="ouro-route-id"),
        ouro_route_org_id: Optional[str] = Header(None, alias="ouro-route-org-id"),
        ouro_route_team_id: Optional[str] = Header(None, alias="ouro-route-team-id"),
        ouro_action_id: Optional[str] = Header(None, alias="ouro-action-id"),
    ):
        import base64
        import os
        import subprocess

        from ouro import Ouro

        ouro = Ouro(api_key=os.environ["OURO_API_KEY"])

        model_name = "crystallm_v1_small"
        prompt_id = uuid4()
        file_name = f"{prompt_id}.txt"
        prompt_path = OUTPUT_PATH / file_name
        model_path = MODELS_PATH / model_name

        # Build the composition string with optional parameters
        composition = request.composition
        command = [
            "python3",
            "/opt/crystallm/bin/make_prompt_file.py",
            composition,
            str(prompt_path),  # Convert Path to string
        ]

        if request.space_group is not None:
            command.extend(["--spacegroup", request.space_group.value])

        print("Running commmand:", command, sep="\n\t")

        ouro.client.post(
            f"/actions/{ouro_action_id}/log",
            json={
                "message": "Generating a prompt file...",
                "asset_id": ouro_route_id,
                "level": "info",
            },
        )

        p = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        # Try to read the prompt file if it was created
        with open(prompt_path, "r") as f:
            prompt = f.read()

        if p.returncode != 0:
            ouro.client.post(
                f"/actions/{ouro_action_id}/log",
                json={
                    "message": "Failed to create prompt file",
                    "asset_id": ouro_route_id,
                    "level": "error",
                },
            )
            return {
                "error": "Failed to create prompt file",
                "return_code": p.returncode,
                "stdout": p.stdout,
                "stderr": p.stderr,
            }

        # Run the model with configurable parameters
        command = [
            "python3",
            "/opt/crystallm/bin/sample.py",
            f"out_dir={model_path}",
            f"start=FILE:{prompt_path}",
            "num_samples=1",
            "top_k=10",
            f"max_new_tokens={request.max_new_tokens}",
            f"temperature={request.temperature}",
            "device=cuda",
            "target=file",
            "compile=true",
        ]

        print("🦙 running commmand:", command, sep="\n\t")
        ouro.client.post(
            f"/actions/{ouro_action_id}/log",
            json={
                "message": "Running the model...",
                "asset_id": ouro_route_id,
                "level": "info",
            },
        )

        try:
            p = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if p.returncode != 0:
                return {
                    "error": "Model execution failed",
                    "stdout": p.stdout,
                    "stderr": p.stderr,
                }

            # List files in the output directory
            print("Output directory contents:", list(OUTPUT_PATH.iterdir()))
            print("Model directory contents:", list(model_path.iterdir()))
            print("repo", os.listdir("/opt/crystallm"))

            volume.commit()

            # Load the output files from our mounted volume - note the path has changed
            sample_path = Path(
                "sample_1.cif"
            )  # This is where sample.py writes the file

            if not sample_path.exists():
                return {
                    "error": f"Output file not found at {sample_path}. Current directory contents: {os.listdir('.')}"
                }

            ouro.client.post(
                f"/actions/{ouro_action_id}/log",
                json={
                    "message": "Created raw CIF file",
                    "asset_id": ouro_route_id,
                    "level": "info",
                },
            )
            ouro.client.post(
                f"/actions/{ouro_action_id}/log",
                json={
                    "message": "Running post-processing...",
                    "asset_id": ouro_route_id,
                    "level": "info",
                },
            )
            # Create directories for raw and processed CIFs
            raw_cifs_dir = OUTPUT_PATH / "raw_cifs"
            processed_cifs_dir = OUTPUT_PATH / "processed_cifs"
            raw_cifs_dir.mkdir(exist_ok=True)
            processed_cifs_dir.mkdir(exist_ok=True)

            # Copy the generated CIF to raw directory instead of moving
            raw_cif_path = raw_cifs_dir / sample_path.name
            import shutil

            shutil.copy2(sample_path, raw_cif_path)
            os.remove(sample_path)  # Clean up the original

            # Run post-processing
            command = [
                "python3",
                "/opt/crystallm/bin/postprocess.py",
                str(raw_cifs_dir),
                str(processed_cifs_dir),
            ]

            print("🔄 Post-processing CIF:", command, sep="\n\t")

            p = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            ouro.client.post(
                f"/actions/{ouro_action_id}/log",
                json={
                    "message": "Successfully processed CIF file",
                    "asset_id": ouro_route_id,
                    "level": "info",
                },
            )

            if p.returncode != 0:
                return {
                    "error": "Post-processing failed",
                    "stdout": p.stdout,
                    "stderr": p.stderr,
                }

            # Read both raw and processed CIFs
            with open(raw_cif_path, "r") as f:
                raw_cif = f.read()

            processed_cif_path = processed_cifs_dir / sample_path.name
            if not processed_cif_path.exists():
                return {
                    "error": "Processed CIF not found",
                    "raw_cif": raw_cif,
                    "processed_cif_path": str(processed_cif_path),
                    "dir_contents": os.listdir(processed_cifs_dir),
                }

            with open(processed_cif_path, "r") as f:
                processed_cif = f.read()

            # Analyze symmetry using pymatgen
            structure = Structure.from_str(processed_cif, fmt="cif")
            spg = SpacegroupAnalyzer(structure, symprec=1e-3)
            symmetry_info = {
                "space_group_number": spg.get_space_group_number(),
                "space_group_symbol": spg.get_space_group_symbol(),
                "crystal_system": spg.get_crystal_system(),
                "point_group": spg.get_point_group_symbol(),
                "is_centrosymmetric": spg.is_laue(),
            }

            # Create a detailed description
            description = (
                f"{request.composition} "
                f"(Space group: {symmetry_info['space_group_symbol']} #{symmetry_info['space_group_number']}, "
                f"Crystal system: {symmetry_info['crystal_system']}, "
                f"Point group: {symmetry_info['point_group']})"
            )

            # Commit any changes to the volume
            volume.commit()

            cif_string = str(processed_cif)
            cif64 = base64.b64encode(cif_string.encode("utf-8")).decode("utf-8")

            return {
                "raw_cif": raw_cif,
                "processed_cif": processed_cif,
                "file": {
                    "name": request.composition,
                    "description": description,
                    "filename": f"{request.composition}.cif",
                    "type": "text/cif",
                    "extension": "cif",
                    "base64": cif64,
                    "org_id": ouro_route_org_id,
                    "team_id": ouro_route_team_id,
                },
            }

        except Exception as e:
            return {"error": str(e), "type": str(type(e))}

    return web_app


# @app.cls(volumes={str(CONTAINER_PATH): volume}, image=image, gpu="T4")
# class CrystaLLM:
#     @modal.enter()
#     def startup(self):
#         from datetime import datetime, timezone

#         print("🏁 Starting up!")
#         self.start_time = datetime.now(timezone.utc)

#         # Ensure output directory exists
#         OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# Nvidia A10G
# $0.000306 / sec
# Nvidia L4
# $0.000222 / sec
# Nvidia T4
# $0.000164 / sec

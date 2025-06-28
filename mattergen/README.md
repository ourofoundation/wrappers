# MatterGen Modal App

This is a Modal app that exposes MatterGen's chemical system conditioned generation capabilities via a REST API. It uses the pre-trained `chemical_system` model from MatterGen to generate crystal structures for a given chemical system.

## Setup

1. Install Modal CLI:
```bash
pip install modal
```

2. Configure Modal:
```bash
modal token new
```

3. Create a secret for MatterGen:
```bash
modal secret create mattergen OURO_API_KEY=<your-ouro-api-key>
```

## Usage

1. Deploy the app:
```bash
modal deploy app.py
```

2. The app exposes two endpoints:

- `GET /`: Welcome message
- `POST /generate`: Generate crystal structures

### Generation Endpoint

The `/generate` endpoint accepts the following parameters:

```json
{
  "chemical_system": "Li-O",  // Required: Chemical system to generate (e.g., "Li-O" or "Fe-Co-Ni")
  "batch_size": 16,          // Optional: Number of structures to generate in each batch (default: 16)
  "num_batches": 1,          // Optional: Number of batches to generate (default: 1)
  "guidance_factor": 2.0     // Optional: Diffusion guidance factor (default: 2.0)
}
```

The endpoint returns a CIF file containing the generated crystal structure.

Example using curl:
```bash
curl -X POST "https://<your-modal-app-url>/generate" \
  -H "Content-Type: application/json" \
  -d '{"chemical_system": "Li-O"}'
```

## Development

The app uses a Modal volume named `mattergen-data` to store model weights. The model is automatically downloaded from Hugging Face when first used.

To run the app locally:
```bash
modal serve app.py
```

## References

- [MatterGen GitHub](https://github.com/microsoft/mattergen)
- [MatterGen Paper](https://www.nature.com/articles/s41586-025-08628-5) 
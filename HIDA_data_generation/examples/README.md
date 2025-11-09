# HIDA Data Generation Examples

This directory contains example scripts demonstrating different ways to use the HIDA data generation pipeline.

## Examples

### Minimal Example (`minimal_example.sh`)

A complete end-to-end example that generates a small dataset for testing.

```bash
cd examples
./minimal_example.sh
```

**What it does:**
- Generates 10 viewpoints using Fibonacci sphere
- Creates 40 procedural textures (10 per category)
- Generates 10 Shapegen objects
- Renders each object from 10 viewpoints
- Uses 4 lights with random intensities

---

## Running Examples

All examples are self-contained and can be run independently:

```bash
cd HIDA_data_generation/examples
./minimal_example.sh
```

## Creating Your Own Examples

Use these examples as templates for your own workflows. Key steps:

1. **Generate utilities** (viewpoints + textures)
2. **Generate objects** (Shapegen and/or Primigen)
3. **Render objects** (using render scripts)
4. **(Optional) Add backgrounds** (using background_adding.py)

## Tips

- Start with minimal example to verify your setup
- Increase sizes gradually (10 → 50 → 100 objects)
- Use `--preview` mode for Shapegen to see all variations
- Monitor disk space (renders can be large)

## Getting Help

- See main [README](../README.md) for detailed documentation
- Check [quickstart.sh](../quickstart.sh) for preset configurations
- Use [hida_generate.py](../hida_generate.py) for unified pipeline

import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Convert a MACE model from GPU to CPU")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input .model file saved with CUDA tensors"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the CPU-only version (default: input_cpu.model)"
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.replace(".model", "_cpu.model")

    if not os.path.isfile(input_path):
        print(f"✗ Error: File '{input_path}' not found.")
        return

    print(f"🔄 Loading model from: {input_path}")
    model = torch.load(input_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

    print("⚙️  Moving model to CPU...")
    model_cpu = model.to("cpu")

    print(f"💾 Saving CPU-only model to: {output_path}")
    torch.save(model_cpu, output_path)

    print("✅ Conversion complete.")

if __name__ == "__main__":
    main()

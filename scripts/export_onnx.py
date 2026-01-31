#!/usr/bin/env python3
"""
ONNX Export Script for CodeTurtle.

Converts the PyTorch CodeBERT model to ONNX format for accelerated inference.
Uses Hugging Face Optimum to handle the conversion and validation.
"""

import argparse
import sys
from pathlib import Path
import logging
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DEFAULT_EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_model(model_name: str, output_dir: Path, quantize: bool = False, quant_config: str = "avx2"):
    """
    Export model to ONNX format.
    
    Args:
        model_name: Hugging Face model ID (e.g. microsoft/codebert-base)
        output_dir: Directory to save the ONNX model
        quantize: Whether to apply dynamic quantization (int8)
    """
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("Dependencies missing. Run: uv sync --all-extras")
        sys.exit(1)

    logger.info(f"🚀 Starting export for {model_name}...")
    start_time = time.time()
    
    # 1. Export
    # export=True triggers the conversion from PyTorch
    logger.info("   Loading and tracing model graph (this may take a minute)...")
    model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Save
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"   Saving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 3. Quantization (Optional)
    if quantize:
        logger.info(f"   Applying dynamic quantization (int8) targetting {quant_config}...")
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from optimum.onnxruntime import ORTQuantizer
        
        quantizer = ORTQuantizer.from_pretrained(model)
        
        dqconfig = None
        if quant_config == "avx512":
             dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        elif quant_config == "arm64":
             dqconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
        else:
             # default safest
             dqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
        
        quantized_path = output_dir / "model_quantized.onnx"
        quantizer.quantize(
            save_dir=output_dir,
            quantization_config=dqconfig,
        )
        logger.info(f"   Quantized model saved to {quantized_path}")

    elapsed = time.time() - start_time
    logger.info(f"✅ Export complete in {elapsed:.1f}s")
    logger.info(f"   Model saved at: {output_dir / 'model.onnx'}")


def main():
    parser = argparse.ArgumentParser(description="Export CodeBERT to ONNX")
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Model to export (default: {DEFAULT_EMBEDDING_MODEL})"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("data/models/codebert_onnx"),
        help="Output directory"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic quantization (int8)"
    )
    parser.add_argument(
        "--quantization-config",
        type=str,
        default="avx2",
        choices=["avx2", "avx512", "arm64"],
        help="Target architecture for quantization (default: avx2)"
    )
    
    args = parser.parse_args()
    
    export_model(args.model, args.output, args.quantize, args.quantization_config)


if __name__ == "__main__":
    main()

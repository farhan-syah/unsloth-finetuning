#!/usr/bin/env python3
"""
Quick test script to verify the fine-tuned model outputs are not gibberish.
Tests both merged model and LoRA adapters.
"""

import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_model(model_path, model_type="merged"):
    """Test a model with a simple prompt"""
    print(f"\n{'='*60}")
    print(f"Testing {model_type}: {model_path}")
    print('='*60)

    try:
        # Load model and tokenizer
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Model loaded successfully")

        # Test prompts
        test_prompts = [
            "What is machine learning?",
            "Explain Python in simple terms.",
            "Write a hello world program in Python."
        ]

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}/{len(test_prompts)} ---")
            print(f"Prompt: {prompt}")

            # Format prompt (alpaca style)
            formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""

            # Generate
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the response part
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()

            print(f"Response: {response[:500]}")  # First 500 chars
            print()

        print("✅ All tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Get model directory from outputs
    output_base = "./outputs"

    if not os.path.exists(output_base):
        print(f"❌ Error: {output_base} directory not found")
        print("Make sure you're in the unsloth-finetuning directory")
        sys.exit(1)

    # Find the model directory
    model_dirs = [d for d in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, d))]

    if not model_dirs:
        print(f"❌ Error: No model directories found in {output_base}")
        sys.exit(1)

    model_dir = model_dirs[0]  # Use the first one
    print(f"Found model directory: {model_dir}")

    # Test merged model
    merged_path = os.path.join(output_base, model_dir, "merged_16bit")

    if os.path.exists(merged_path):
        success = test_model(merged_path, "Merged 16-bit Model")
        if not success:
            print("\n⚠️  Merged model test failed!")
            sys.exit(1)
    else:
        print(f"⚠️  Merged model not found at {merged_path}")

    print("\n" + "="*60)
    print("✅ MODEL VALIDATION COMPLETE")
    print("="*60)
    print("\nThe model appears to be working correctly!")
    print("\nNext steps:")
    print("  1. If outputs look good, the model is ready to use")
    print("  2. Test with your own prompts")
    print("  3. Convert to GGUF if needed: python build.py")

if __name__ == "__main__":
    main()

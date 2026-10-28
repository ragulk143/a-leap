import os
import sys
import argparse
from aleap.api.server import start_server

def main():
    print("="*60)
    print("               A-LEAP PRODUCTION RUNTIME ENGINE              ")
    print("="*60)

    # Setup core argument routing rules
    parser = argparse.ArgumentParser(description="A-LEAP Inference Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # Register the 'serve' command pipeline
    serve_parser = subparsers.add_parser("serve", help="Launch the local inference API server")
    serve_parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Absolute or relative path to the local model folder containing .safetensors files"
    )

    # Parse arguments coming from the terminal command line
    args = parser.parse_args()

    if args.command == "serve":
        # Resolve the directory path cleanly and drop it straight into environment memory
        absolute_model_path = os.path.abspath(args.model)
        os.environ["ALEAP_MODEL_PATH"] = absolute_model_path
        
        start_server()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()

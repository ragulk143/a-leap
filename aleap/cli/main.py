import sys
from aleap.api.server import start_server

def main():
    print("="*60)
    print("               A-LEAP PRODUCTION RUNTIME ENGINE              ")
    print("="*60)
    
    # If the user targets the standard execution command layout
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        start_server()
    else:
        print("[*] Usage: aleap serve")

if __name__ == "__main__":
    main()


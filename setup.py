#!/usr/bin/env python3
"""
Setup script for PST Email Extractor.
Verifies dependencies and environment configuration.
"""

import sys
import subprocess
import platform


def check_python_version():
    """Verify Python version is 3.6 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 6):
        print(f"[!] Python 3.6+ required (found {version.major}.{version.minor}.{version.micro})")
        return False
    print(f"[+] Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_pip():
    """Verify pip is available."""
    try:
        import pip
        print(f"[+] pip is installed")
        return True
    except ImportError:
        print("[!] pip is not installed")
        return False


def install_dependencies():
    """Install required Python packages."""
    print("\n[*] Installing Python dependencies...")
    try:
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            "requirements.txt"
        ])
        print("[+] Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] Failed to install dependencies: {e}")
        return False


def check_pypff():
    """Check if pypff can be imported."""
    try:
        import pypff
        print("[+] pypff is installed and importable")
        return True
    except ImportError:
        print("[!] pypff not found")
        print("\n[*] pypff requires system dependencies:")
        
        os_name = platform.system()
        if os_name == "Windows":
            print("    Windows: May require precompiled binaries or WSL")
            print("    See: https://github.com/libyal/libpff/releases")
        elif os_name == "Linux":
            print("    Linux: sudo apt-get install libpff-dev python3-dev")
        elif os_name == "Darwin":
            print("    macOS: brew install libpff")
        
        return False


def run_tests():
    """Run module tests."""
    print("\n[*] Running module tests...")
    try:
        result = subprocess.run(
            [sys.executable, "test_modules.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("[+] All tests passed")
            return True
        else:
            print("[!] Some tests failed")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"[!] Failed to run tests: {e}")
        return False


def main():
    """Run setup verification."""
    print("="*60)
    print("PST Email Extractor - Setup")
    print("="*60 + "\n")
    
    checks = []
    
    # basic environment checks
    print("[*] Checking environment...")
    checks.append(("Python Version", check_python_version()))
    checks.append(("pip", check_pip()))
    
    # install dependencies
    if checks[-1][1]:  # if pip is available
        checks.append(("Dependencies", install_dependencies()))
    
    # check pypff
    checks.append(("pypff", check_pypff()))
    
    # run tests
    checks.append(("Module Tests", run_tests()))
    
    # summary
    print("\n" + "="*60)
    print("Setup Summary")
    print("="*60)
    
    for check_name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status:8} - {check_name}")
    
    print("="*60 + "\n")
    
    # final verdict
    all_passed = all(result[1] for result in checks)
    
    if all_passed:
        print("[+] Setup complete! Ready to use.")
        print("\nUsage:")
        print("  python main.py -i emails.pst -o output/ -j -c")
        print("\nFor more info:")
        print("  python main.py --help")
        print("  cat USAGE.md")
        return 0
    else:
        print("[!] Setup incomplete - some checks failed")
        print("\nReview error messages above and:")
        print("  1. Install missing system dependencies")
        print("  2. Re-run this setup script")
        print("\nSee USAGE.md for detailed installation instructions")
        return 1


if __name__ == "__main__":
    sys.exit(main())


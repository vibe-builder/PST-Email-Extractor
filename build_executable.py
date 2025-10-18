#!/usr/bin/env python3
"""
Build script to create Windows executable from PyQt6 GUI application.
Uses PyInstaller to package the application as a standalone .exe
"""

import subprocess
import sys
import os
from pathlib import Path


def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        print(f"[+] PyInstaller found: {PyInstaller.__version__}")
        return True
    except ImportError:
        print("[!] PyInstaller not found")
        print("[*] Install with: pip install pyinstaller")
        return False


def create_spec_file():
    """Create PyInstaller spec file."""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('pst_tools', 'pst_tools'),
    ],
    hiddenimports=['pypff', 'unicodecsv', 'PyQt6', 'PyQt6.sip'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PST_Email_Extractor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # no console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # add .ico file here if you have one
)
'''

    with open('pst_extractor.spec', 'w') as f:
        f.write(spec_content)

    print("[+] Created pst_extractor.spec")


def build_executable():
    """Build the executable using PyInstaller."""
    print("\n[*] Building Windows executable...")
    print("[*] This may take a few minutes...\n")

    try:
        # run pyinstaller with the spec file
        result = subprocess.run(
            [sys.executable, '-m', 'PyInstaller', 'pst_extractor.spec', '--clean'],
            check=True,
            capture_output=True,
            text=True
        )

        print(result.stdout)

        # check if executable was created
        exe_path = Path('dist') / 'PST_Email_Extractor.exe'
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"\n[+] Build successful!")
            print(f"[+] Executable: {exe_path.absolute()}")
            print(f"[+] Size: {size_mb:.1f} MB")
            print("\n[*] You can now distribute PST_Email_Extractor.exe")
            return True
        else:
            print("[!] Build completed but executable not found")
            return False

    except subprocess.CalledProcessError as e:
        print(f"[!] Build failed: {e}")
        print(e.stderr)
        return False


def main():
    """Main build process."""
    print("="*60)
    print("PST Email Extractor - Executable Builder")
    print("="*60 + "\n")

    # check prerequisites
    if not check_pyinstaller():
        print("\n[*] Installing PyInstaller...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])

    # check if gui.py exists
    if not Path('gui.py').exists():
        print("[!] gui.py not found in current directory")
        return 1

    # create spec file
    create_spec_file()

    # build executable
    if build_executable():
        print("\n" + "="*60)
        print("Build Complete!")
        print("="*60)
        return 0
    else:
        print("\n[!] Build failed - check errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())


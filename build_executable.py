#!/usr/bin/env python3
"""
Build script to create executable from the CustomTkinter GUI application.
Uses PyInstaller to package the application as a standalone executable.
Supports Windows, macOS, and Linux builds.
"""

import subprocess
import sys
from pathlib import Path


def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import importlib.util
        if importlib.util.find_spec("PyInstaller") is None:
            raise ImportError("PyInstaller not found")

        import PyInstaller  # type: ignore
        print(f"[+] PyInstaller found: {PyInstaller.__version__}")
        return True
    except ImportError:
        print("[!] PyInstaller not found")
        print("[!] Install it manually or use: pip install -e '.[build]'")
        print("[!] Or run: pip install pyinstaller")
        print("[!] This script will not auto-install dependencies for security reasons.")
        return False


def create_spec_file():
    """Create PyInstaller spec file with CustomTkinter configuration."""
    import platform

    # Determine platform-specific settings
    is_windows = platform.system() == "Windows"
    is_macos = platform.system() == "Darwin"

    # Platform-specific executable name
    exe_name = 'PST_Email_Extractor'
    if is_windows:
        exe_name += '.exe'
    elif is_macos:
        exe_name += '.app'

    # Platform-specific data paths
    data_sep = '\\' if is_windows else '/'
    data_paths = [
        (f'src{data_sep}pst_email_extractor{data_sep}ai{data_sep}data', 'pst_email_extractor/ai/data'),
        (f'src{data_sep}pst_email_extractor{data_sep}ai{data_sep}lib', 'pst_email_extractor/ai/lib'),
        (f'src{data_sep}pst_email_extractor{data_sep}ai{data_sep}model_dir', 'pst_email_extractor/ai/model_dir'),
        (f'src{data_sep}pst_email_extractor', 'pst_email_extractor'),
    ]

    # Add logo if available
    icon_param = ""
    if Path('logo.ico').exists() and is_windows:
        icon_param = ", icon='logo.ico'"

    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file generated with uv support

block_cipher = None

a = Analysis(
    ['src/pst_email_extractor/gui/main.py'],
    pathex=[],
    binaries=[],
    datas={data_paths},
    # Hidden imports required by runtime (tkinter is discovered from stdlib)
    hiddenimports=[
        'pypff',
        'polars',  # Memory-efficient DataFrame library
        'pst_email_extractor',
        'pst_email_extractor.cli.app',
        'pst_email_extractor.core.extraction',
        'pst_email_extractor.core.models',
        'pst_email_extractor.core.backends.pypff',
        'pst_email_extractor.exporters',
        'customtkinter',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'matplotlib',  # Not needed for this app
        'PySide6',
        'shiboken6',
    ],
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
    name='{exe_name}',
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
    entitlements_file=None{icon_param},
)
'''

    spec_file = 'pst_extractor.spec'
    with open(spec_file, 'w', encoding='utf-8') as f:
        f.write(spec_content)

    print(f"[+] Created {spec_file} with modern PySide6 configuration")


def build_executable():
    """Build the executable using PyInstaller with uv optimizations."""
    import platform
    platform_name = platform.system()
    print(f"\n[*] Building {platform_name} executable...")
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
        import platform
        is_windows = platform.system() == "Windows"
        is_macos = platform.system() == "Darwin"

        exe_name = 'PST_Email_Extractor'
        if is_windows:
            exe_name += '.exe'
        elif is_macos:
            exe_name += '.app'

        exe_path = Path('dist') / exe_name
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print("\n[+] Build successful!")
            print(f"[+] Executable: {exe_path.absolute()}")
            print(f"[+] Size: {size_mb:.1f} MB")
            print(f"\n[*] You can now distribute {exe_name}")
            return True
        else:
            print("[!] Build completed but executable not found")
            return False

    except subprocess.CalledProcessError as e:
        print(f"[!] Build failed: {e}")
        print(e.stderr)
        return False


def install_dependencies():
    """Install project dependencies using pip."""
    print("[*] Installing project dependencies with pip...")

    try:
        # Install main dependencies
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], check=True)
        print("[+] Main dependencies installed")

        # Install build dependencies
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyinstaller'], check=True)
        print("[+] Build dependencies installed")

        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] Failed to install dependencies with pip: {e}")
        return False


def main():
    """Main build process."""
    print("="*60)
    print("PST Email Extractor - Executable Builder")
    print("Using PyInstaller for cross-platform executable creation")
    print("="*60 + "\n")

    # Install dependencies
    if not install_dependencies():
        print("[!] Dependency installation failed")
        return 1

    # Check PyInstaller
    if not check_pyinstaller():
        return 1

    # Check if GUI entry point exists
    gui_entry = Path('src/pst_email_extractor/gui/main.py')
    if not gui_entry.exists():
        print(f"[!] GUI entry point not found: {gui_entry}")
        return 1

    # Create spec file
    create_spec_file()

    # Build executable
    if build_executable():
        print("\n" + "="*60)
        print("Build Complete!")
        print("Executable created with modern PySide6")
        print("="*60)
        return 0
    else:
        print("\n[!] Build failed - check errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())



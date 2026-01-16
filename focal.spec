# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Focal.

Build with:
    pyinstaller focal.spec

Output will be in dist/Focal.app (macOS) or dist/Focal (Windows/Linux)
"""

import sys
from pathlib import Path

block_cipher = None

# Project paths
PROJECT_ROOT = Path(SPECPATH)
SRC_DIR = PROJECT_ROOT / "src"
ASSETS_DIR = PROJECT_ROOT / "assets"
ICON_PNG = ASSETS_DIR / "icons" / "focal-icon-1.png"
ICON_ICNS = ASSETS_DIR / "icons" / "focal-icon.icns"  # For macOS, convert PNG first
ICON_ICO = ASSETS_DIR / "icons" / "focal-icon.ico"    # For Windows, convert PNG first

# Determine icon based on platform
if sys.platform == "darwin":
    icon_file = str(ICON_ICNS) if ICON_ICNS.exists() else str(ICON_PNG)
elif sys.platform == "win32":
    icon_file = str(ICON_ICO) if ICON_ICO.exists() else None
else:
    icon_file = str(ICON_PNG) if ICON_PNG.exists() else None

a = Analysis(
    [str(SRC_DIR / "focal" / "main.py")],
    pathex=[str(SRC_DIR)],
    binaries=[],
    datas=[],
    hiddenimports=[
        # NumPy imports that PyInstaller sometimes misses
        "numpy",
        "numpy.lib.format",
        # OpenCV
        "cv2",
        # Numba (JIT compiler) - often has hidden imports
        "numba",
        "numba.core.types",
        # PySide6 plugins
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unused Qt modules to reduce size
        "PySide6.QtWebEngine",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtNetwork",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.Qt3D",
        "PySide6.QtBluetooth",
        "PySide6.QtMultimedia",
        "PySide6.QtPositioning",
        "PySide6.QtSensors",
        "PySide6.QtSerialPort",
        "PySide6.QtSql",
        "PySide6.QtTest",
        "PySide6.QtXml",
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
    [],
    exclude_binaries=True,
    name="Focal",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Focal",
)

# macOS-specific: Create .app bundle
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Focal.app",
        icon=icon_file,
        bundle_identifier="com.focal.app",
        info_plist={
            "CFBundleName": "Focal",
            "CFBundleDisplayName": "Focal",
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "0.1.0",
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,  # Support dark mode
        },
    )

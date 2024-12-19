# -*- mode: python ; coding: utf-8 -*-

import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

a = Analysis(
    ['ecare.py'],
    pathex=[
    './lib/convert.py',
    './lib/distance.py',
    './lib/face.py',
    './lib/find.py',
    './lib/idlist.py',
    './lib/movement.py',
    './lib/preload.py',
    './lib/show.py'],
    binaries=[],
    datas=[('D:/Pythoncode/e-care/asserts', 'asserts')],
    hiddenimports=[
    'numpy',
    'matplotlib',
    'mediapipe',
    'cv2',
    'scipy',
    'tqdm',
    'ffmpeg',
    'dataclasses',
    'setuptools'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ecare',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

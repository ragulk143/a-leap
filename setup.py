import os
import sys
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the native C++/CUDA extensions.")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Ensure target location matches internal python namespaces
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release"
        ]

        build_args = ["--config", "Release", "--", "-j4"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        try:
            print("\n" + "="*70)
            print("[A-LEAP Build] Compiling high-performance native C++/CUDA extensions...")
            print("="*70)
            
            # Run CMake config and build pipelines
            subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
            subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)
            print("\n[A-LEAP Build] Native C++/CUDA acceleration compiled successfully!")
            print("="*70 + "\n")
        except Exception as e:
            print("\n" + "!"*70)
            print(f"[!] WARNING: Native compilation failed: {e}")
            print("[!] Falling back to lightweight functional mode.")
            print("[!] The CLI serving engine and API server will still function normally.")
            print("!"*70 + "\n")

setup(
    name="a-leap",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[CMakeExtension("aleap_api")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)

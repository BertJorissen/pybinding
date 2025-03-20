import os
import sys
import shutil
import platform

from subprocess import check_call, CalledProcessError

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.egg_info import manifest_maker


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + os.environ.get("PB_CMAKE_LIBRARY_OUTPUT_DIRECTORY", extdir),
                      "-DPython_EXECUTABLE=" + os.environ.get("PB_PYTHON_EXECUTABLE", sys.executable)]
        cmake_args += ["-DPB_WERROR=" + os.environ.get("PB_WERROR", "OFF"),
                       "-DPB_TESTS=" + os.environ.get("PB_TESTS", "OFF"),
                       "-DPB_NATIVE_SIMD=" + os.environ.get("PB_NATIVE_SIMD", "ON"),
                       "-DPB_MKL=" + os.environ.get("PB_MKL", "OFF"),
                       "-DPB_CUDA=" + os.environ.get("PB_CUDA", "OFF"),
                       "-DPB_CARTESIAN_FLOAT=" + os.environ.get("PB_CARTESIAN_FLOAT", "OFF")]

        cfg = os.environ.get("PB_BUILD_TYPE", "Release")
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)]
            cmake_args += ["-A", "x64" if sys.maxsize > 2**32 else "Win32"]
            build_args += ["--", "/v:m", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            if "-j" not in os.environ.get("MAKEFLAGS", ""):
                parallel_jobs = int(os.cpu_count()) if not os.environ.get("READTHEDOCS") else 1
                build_args += ["--", "-j{}".format(parallel_jobs)]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DCPB_VERSION=\\"{}\\"'.format(env.get("CXXFLAGS", ""),
                                                             self.distribution.get_version())

        def build():
            os.makedirs(self.build_temp, exist_ok=True)
            check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
            check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

        try:
            build()
        except CalledProcessError:  # possible CMake error if the build cache has been copied
            shutil.rmtree(self.build_temp)  # delete build cache and try again
            build()


manifest_maker.template = "setup.manifest"
setup(
    packages=find_packages(exclude=['cppcore', 'cppmodule', 'tests']) +
             ['pybinding.tests',
              'pybinding.tests.baseline_data.kpm',
              'pybinding.tests.baseline_data.lattice',
              'pybinding.tests.baseline_data.parallel',
              'pybinding.tests.baseline_data.shape',
              'pybinding.tests.baseline_data.solver',
              'pybinding.tests.baseline_data.system',
              'pybinding.tests.baseline_plots.results',
              'pybinding.tests.baseline_plots.system',
              'pybinding.tests.utils'],
    package_dir={'pybinding.tests': 'tests'},
    include_package_data=True,
    ext_modules=[CMakeExtension('_pybinding')],
    cmdclass=dict(build_ext=CMakeBuild)
)

# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/philipp/entw/gpulab/superresolution_task

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/philipp/entw/gpulab/superresolution_task

# Include any dependencies generated for this target.
include CMakeFiles/superresolution.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/superresolution.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/superresolution.dir/flags.make

CMakeFiles/superresolution.dir/src/superresolution_main.o: CMakeFiles/superresolution.dir/flags.make
CMakeFiles/superresolution.dir/src/superresolution_main.o: src/superresolution_main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/philipp/entw/gpulab/superresolution_task/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/superresolution.dir/src/superresolution_main.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/superresolution.dir/src/superresolution_main.o -c /home/philipp/entw/gpulab/superresolution_task/src/superresolution_main.cpp

CMakeFiles/superresolution.dir/src/superresolution_main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/superresolution.dir/src/superresolution_main.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/philipp/entw/gpulab/superresolution_task/src/superresolution_main.cpp > CMakeFiles/superresolution.dir/src/superresolution_main.i

CMakeFiles/superresolution.dir/src/superresolution_main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/superresolution.dir/src/superresolution_main.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/philipp/entw/gpulab/superresolution_task/src/superresolution_main.cpp -o CMakeFiles/superresolution.dir/src/superresolution_main.s

CMakeFiles/superresolution.dir/src/superresolution_main.o.requires:
.PHONY : CMakeFiles/superresolution.dir/src/superresolution_main.o.requires

CMakeFiles/superresolution.dir/src/superresolution_main.o.provides: CMakeFiles/superresolution.dir/src/superresolution_main.o.requires
	$(MAKE) -f CMakeFiles/superresolution.dir/build.make CMakeFiles/superresolution.dir/src/superresolution_main.o.provides.build
.PHONY : CMakeFiles/superresolution.dir/src/superresolution_main.o.provides

CMakeFiles/superresolution.dir/src/superresolution_main.o.provides.build: CMakeFiles/superresolution.dir/src/superresolution_main.o

# Object files for target superresolution
superresolution_OBJECTS = \
"CMakeFiles/superresolution.dir/src/superresolution_main.o"

# External object files for target superresolution
superresolution_EXTERNAL_OBJECTS =

bin/superresolution: CMakeFiles/superresolution.dir/src/superresolution_main.o
bin/superresolution: lib/libsuperresolutionlib.so
bin/superresolution: lib/libsuperresolutionlibGPU.so
bin/superresolution: lib/libflowlib.so
bin/superresolution: lib/libflowlibGPU.so
bin/superresolution: lib/libimagepyramid.so
bin/superresolution: lib/libimagepyramidGPU.so
bin/superresolution: lib/liblinearoperations.so
bin/superresolution: lib/liblinearoperationsGPU.so
bin/superresolution: lib/libfilesystem.so
bin/superresolution: lib/libauxiliaryCPU.so
bin/superresolution: lib/libauxiliaryGPU.so
bin/superresolution: /usr/local/cuda/lib/libcudart.so
bin/superresolution: /usr/lib/libcuda.so
bin/superresolution: CMakeFiles/superresolution.dir/build.make
bin/superresolution: CMakeFiles/superresolution.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable bin/superresolution"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/superresolution.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/superresolution.dir/build: bin/superresolution
.PHONY : CMakeFiles/superresolution.dir/build

CMakeFiles/superresolution.dir/requires: CMakeFiles/superresolution.dir/src/superresolution_main.o.requires
.PHONY : CMakeFiles/superresolution.dir/requires

CMakeFiles/superresolution.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/superresolution.dir/cmake_clean.cmake
.PHONY : CMakeFiles/superresolution.dir/clean

CMakeFiles/superresolution.dir/depend:
	cd /home/philipp/entw/gpulab/superresolution_task && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/philipp/entw/gpulab/superresolution_task /home/philipp/entw/gpulab/superresolution_task /home/philipp/entw/gpulab/superresolution_task /home/philipp/entw/gpulab/superresolution_task /home/philipp/entw/gpulab/superresolution_task/CMakeFiles/superresolution.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/superresolution.dir/depend

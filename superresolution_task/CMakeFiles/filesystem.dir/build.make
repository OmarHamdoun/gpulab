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
include CMakeFiles/filesystem.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/filesystem.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/filesystem.dir/flags.make

CMakeFiles/filesystem.dir/src/filesystem/filesystem.o: CMakeFiles/filesystem.dir/flags.make
CMakeFiles/filesystem.dir/src/filesystem/filesystem.o: src/filesystem/filesystem.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/philipp/entw/gpulab/superresolution_task/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/filesystem.dir/src/filesystem/filesystem.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/filesystem.dir/src/filesystem/filesystem.o -c /home/philipp/entw/gpulab/superresolution_task/src/filesystem/filesystem.cpp

CMakeFiles/filesystem.dir/src/filesystem/filesystem.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/filesystem.dir/src/filesystem/filesystem.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/philipp/entw/gpulab/superresolution_task/src/filesystem/filesystem.cpp > CMakeFiles/filesystem.dir/src/filesystem/filesystem.i

CMakeFiles/filesystem.dir/src/filesystem/filesystem.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/filesystem.dir/src/filesystem/filesystem.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/philipp/entw/gpulab/superresolution_task/src/filesystem/filesystem.cpp -o CMakeFiles/filesystem.dir/src/filesystem/filesystem.s

CMakeFiles/filesystem.dir/src/filesystem/filesystem.o.requires:
.PHONY : CMakeFiles/filesystem.dir/src/filesystem/filesystem.o.requires

CMakeFiles/filesystem.dir/src/filesystem/filesystem.o.provides: CMakeFiles/filesystem.dir/src/filesystem/filesystem.o.requires
	$(MAKE) -f CMakeFiles/filesystem.dir/build.make CMakeFiles/filesystem.dir/src/filesystem/filesystem.o.provides.build
.PHONY : CMakeFiles/filesystem.dir/src/filesystem/filesystem.o.provides

CMakeFiles/filesystem.dir/src/filesystem/filesystem.o.provides.build: CMakeFiles/filesystem.dir/src/filesystem/filesystem.o

# Object files for target filesystem
filesystem_OBJECTS = \
"CMakeFiles/filesystem.dir/src/filesystem/filesystem.o"

# External object files for target filesystem
filesystem_EXTERNAL_OBJECTS =

lib/libfilesystem.so: CMakeFiles/filesystem.dir/src/filesystem/filesystem.o
lib/libfilesystem.so: CMakeFiles/filesystem.dir/build.make
lib/libfilesystem.so: CMakeFiles/filesystem.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library lib/libfilesystem.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/filesystem.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/filesystem.dir/build: lib/libfilesystem.so
.PHONY : CMakeFiles/filesystem.dir/build

CMakeFiles/filesystem.dir/requires: CMakeFiles/filesystem.dir/src/filesystem/filesystem.o.requires
.PHONY : CMakeFiles/filesystem.dir/requires

CMakeFiles/filesystem.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/filesystem.dir/cmake_clean.cmake
.PHONY : CMakeFiles/filesystem.dir/clean

CMakeFiles/filesystem.dir/depend:
	cd /home/philipp/entw/gpulab/superresolution_task && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/philipp/entw/gpulab/superresolution_task /home/philipp/entw/gpulab/superresolution_task /home/philipp/entw/gpulab/superresolution_task /home/philipp/entw/gpulab/superresolution_task /home/philipp/entw/gpulab/superresolution_task/CMakeFiles/filesystem.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/filesystem.dir/depend


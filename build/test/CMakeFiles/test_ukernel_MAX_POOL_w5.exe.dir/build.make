# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sshuba/SMaLLFramework

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sshuba/SMaLLFramework/build

# Include any dependencies generated for this target.
include test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/flags.make

test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/kernel_benchmark_test.cpp.o: test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/flags.make
test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/kernel_benchmark_test.cpp.o: /home/sshuba/SMaLLFramework/test/kernel_benchmark_test.cpp
test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/kernel_benchmark_test.cpp.o: test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sshuba/SMaLLFramework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/kernel_benchmark_test.cpp.o"
	cd /home/sshuba/SMaLLFramework/build/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/kernel_benchmark_test.cpp.o -MF CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/kernel_benchmark_test.cpp.o.d -o CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/kernel_benchmark_test.cpp.o -c /home/sshuba/SMaLLFramework/test/kernel_benchmark_test.cpp

test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/kernel_benchmark_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/kernel_benchmark_test.cpp.i"
	cd /home/sshuba/SMaLLFramework/build/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sshuba/SMaLLFramework/test/kernel_benchmark_test.cpp > CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/kernel_benchmark_test.cpp.i

test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/kernel_benchmark_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/kernel_benchmark_test.cpp.s"
	cd /home/sshuba/SMaLLFramework/build/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sshuba/SMaLLFramework/test/kernel_benchmark_test.cpp -o CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/kernel_benchmark_test.cpp.s

# Object files for target test_ukernel_MAX_POOL_w5.exe
test_ukernel_MAX_POOL_w5_exe_OBJECTS = \
"CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/kernel_benchmark_test.cpp.o"

# External object files for target test_ukernel_MAX_POOL_w5.exe
test_ukernel_MAX_POOL_w5_exe_EXTERNAL_OBJECTS =

test/test_ukernel_MAX_POOL_w5.exe: test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/kernel_benchmark_test.cpp.o
test/test_ukernel_MAX_POOL_w5.exe: test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/build.make
test/test_ukernel_MAX_POOL_w5.exe: test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sshuba/SMaLLFramework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_ukernel_MAX_POOL_w5.exe"
	cd /home/sshuba/SMaLLFramework/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/build: test/test_ukernel_MAX_POOL_w5.exe
.PHONY : test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/build

test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/clean:
	cd /home/sshuba/SMaLLFramework/build/test && $(CMAKE_COMMAND) -P CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/clean

test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/depend:
	cd /home/sshuba/SMaLLFramework/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sshuba/SMaLLFramework /home/sshuba/SMaLLFramework/test /home/sshuba/SMaLLFramework/build /home/sshuba/SMaLLFramework/build/test /home/sshuba/SMaLLFramework/build/test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test_ukernel_MAX_POOL_w5.exe.dir/depend


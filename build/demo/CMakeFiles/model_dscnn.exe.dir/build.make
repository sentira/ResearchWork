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
include demo/CMakeFiles/model_dscnn.exe.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include demo/CMakeFiles/model_dscnn.exe.dir/compiler_depend.make

# Include the progress variables for this target.
include demo/CMakeFiles/model_dscnn.exe.dir/progress.make

# Include the compile flags for this target's objects.
include demo/CMakeFiles/model_dscnn.exe.dir/flags.make

demo/CMakeFiles/model_dscnn.exe.dir/dscnn.cpp.o: demo/CMakeFiles/model_dscnn.exe.dir/flags.make
demo/CMakeFiles/model_dscnn.exe.dir/dscnn.cpp.o: /home/sshuba/SMaLLFramework/demo/dscnn.cpp
demo/CMakeFiles/model_dscnn.exe.dir/dscnn.cpp.o: demo/CMakeFiles/model_dscnn.exe.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sshuba/SMaLLFramework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object demo/CMakeFiles/model_dscnn.exe.dir/dscnn.cpp.o"
	cd /home/sshuba/SMaLLFramework/build/demo && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT demo/CMakeFiles/model_dscnn.exe.dir/dscnn.cpp.o -MF CMakeFiles/model_dscnn.exe.dir/dscnn.cpp.o.d -o CMakeFiles/model_dscnn.exe.dir/dscnn.cpp.o -c /home/sshuba/SMaLLFramework/demo/dscnn.cpp

demo/CMakeFiles/model_dscnn.exe.dir/dscnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/model_dscnn.exe.dir/dscnn.cpp.i"
	cd /home/sshuba/SMaLLFramework/build/demo && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sshuba/SMaLLFramework/demo/dscnn.cpp > CMakeFiles/model_dscnn.exe.dir/dscnn.cpp.i

demo/CMakeFiles/model_dscnn.exe.dir/dscnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/model_dscnn.exe.dir/dscnn.cpp.s"
	cd /home/sshuba/SMaLLFramework/build/demo && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sshuba/SMaLLFramework/demo/dscnn.cpp -o CMakeFiles/model_dscnn.exe.dir/dscnn.cpp.s

# Object files for target model_dscnn.exe
model_dscnn_exe_OBJECTS = \
"CMakeFiles/model_dscnn.exe.dir/dscnn.cpp.o"

# External object files for target model_dscnn.exe
model_dscnn_exe_EXTERNAL_OBJECTS =

demo/model_dscnn.exe: demo/CMakeFiles/model_dscnn.exe.dir/dscnn.cpp.o
demo/model_dscnn.exe: demo/CMakeFiles/model_dscnn.exe.dir/build.make
demo/model_dscnn.exe: demo/CMakeFiles/model_dscnn.exe.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sshuba/SMaLLFramework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable model_dscnn.exe"
	cd /home/sshuba/SMaLLFramework/build/demo && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/model_dscnn.exe.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
demo/CMakeFiles/model_dscnn.exe.dir/build: demo/model_dscnn.exe
.PHONY : demo/CMakeFiles/model_dscnn.exe.dir/build

demo/CMakeFiles/model_dscnn.exe.dir/clean:
	cd /home/sshuba/SMaLLFramework/build/demo && $(CMAKE_COMMAND) -P CMakeFiles/model_dscnn.exe.dir/cmake_clean.cmake
.PHONY : demo/CMakeFiles/model_dscnn.exe.dir/clean

demo/CMakeFiles/model_dscnn.exe.dir/depend:
	cd /home/sshuba/SMaLLFramework/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sshuba/SMaLLFramework /home/sshuba/SMaLLFramework/demo /home/sshuba/SMaLLFramework/build /home/sshuba/SMaLLFramework/build/demo /home/sshuba/SMaLLFramework/build/demo/CMakeFiles/model_dscnn.exe.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : demo/CMakeFiles/model_dscnn.exe.dir/depend


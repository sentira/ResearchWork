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
include demo/CMakeFiles/check_interface.o.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include demo/CMakeFiles/check_interface.o.dir/compiler_depend.make

# Include the progress variables for this target.
include demo/CMakeFiles/check_interface.o.dir/progress.make

# Include the compile flags for this target's objects.
include demo/CMakeFiles/check_interface.o.dir/flags.make

demo/CMakeFiles/check_interface.o.dir/check_interface_abstract.cpp.o: demo/CMakeFiles/check_interface.o.dir/flags.make
demo/CMakeFiles/check_interface.o.dir/check_interface_abstract.cpp.o: /home/sshuba/SMaLLFramework/demo/check_interface_abstract.cpp
demo/CMakeFiles/check_interface.o.dir/check_interface_abstract.cpp.o: demo/CMakeFiles/check_interface.o.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sshuba/SMaLLFramework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object demo/CMakeFiles/check_interface.o.dir/check_interface_abstract.cpp.o"
	cd /home/sshuba/SMaLLFramework/build/demo && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT demo/CMakeFiles/check_interface.o.dir/check_interface_abstract.cpp.o -MF CMakeFiles/check_interface.o.dir/check_interface_abstract.cpp.o.d -o CMakeFiles/check_interface.o.dir/check_interface_abstract.cpp.o -c /home/sshuba/SMaLLFramework/demo/check_interface_abstract.cpp

demo/CMakeFiles/check_interface.o.dir/check_interface_abstract.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/check_interface.o.dir/check_interface_abstract.cpp.i"
	cd /home/sshuba/SMaLLFramework/build/demo && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sshuba/SMaLLFramework/demo/check_interface_abstract.cpp > CMakeFiles/check_interface.o.dir/check_interface_abstract.cpp.i

demo/CMakeFiles/check_interface.o.dir/check_interface_abstract.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/check_interface.o.dir/check_interface_abstract.cpp.s"
	cd /home/sshuba/SMaLLFramework/build/demo && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sshuba/SMaLLFramework/demo/check_interface_abstract.cpp -o CMakeFiles/check_interface.o.dir/check_interface_abstract.cpp.s

# Object files for target check_interface.o
check_interface_o_OBJECTS = \
"CMakeFiles/check_interface.o.dir/check_interface_abstract.cpp.o"

# External object files for target check_interface.o
check_interface_o_EXTERNAL_OBJECTS =

demo/libcheck_interface.o.a: demo/CMakeFiles/check_interface.o.dir/check_interface_abstract.cpp.o
demo/libcheck_interface.o.a: demo/CMakeFiles/check_interface.o.dir/build.make
demo/libcheck_interface.o.a: demo/CMakeFiles/check_interface.o.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sshuba/SMaLLFramework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libcheck_interface.o.a"
	cd /home/sshuba/SMaLLFramework/build/demo && $(CMAKE_COMMAND) -P CMakeFiles/check_interface.o.dir/cmake_clean_target.cmake
	cd /home/sshuba/SMaLLFramework/build/demo && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/check_interface.o.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
demo/CMakeFiles/check_interface.o.dir/build: demo/libcheck_interface.o.a
.PHONY : demo/CMakeFiles/check_interface.o.dir/build

demo/CMakeFiles/check_interface.o.dir/clean:
	cd /home/sshuba/SMaLLFramework/build/demo && $(CMAKE_COMMAND) -P CMakeFiles/check_interface.o.dir/cmake_clean.cmake
.PHONY : demo/CMakeFiles/check_interface.o.dir/clean

demo/CMakeFiles/check_interface.o.dir/depend:
	cd /home/sshuba/SMaLLFramework/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sshuba/SMaLLFramework /home/sshuba/SMaLLFramework/demo /home/sshuba/SMaLLFramework/build /home/sshuba/SMaLLFramework/build/demo /home/sshuba/SMaLLFramework/build/demo/CMakeFiles/check_interface.o.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : demo/CMakeFiles/check_interface.o.dir/depend


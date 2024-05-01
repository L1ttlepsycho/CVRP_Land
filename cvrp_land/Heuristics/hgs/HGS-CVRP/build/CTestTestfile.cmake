# CMake generated Testfile for 
# Source directory: /home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0
# Build directory: /home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(bin_test_X-n101-k25 "/usr/bin/cmake" "-DINSTANCE=X-n101-k25" "-DCOST=27591" "-DROUND=1" "-P" "/home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0/Test/TestExecutable.cmake")
set_tests_properties(bin_test_X-n101-k25 PROPERTIES  _BACKTRACE_TRIPLES "/home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0/CMakeLists.txt;28;add_test;/home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0/CMakeLists.txt;0;")
add_test(bin_test_X-n106-k14 "/usr/bin/cmake" "-DINSTANCE=X-n110-k13" "-DCOST=14971" "-DROUND=1" "-P" "/home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0/Test/TestExecutable.cmake")
set_tests_properties(bin_test_X-n106-k14 PROPERTIES  _BACKTRACE_TRIPLES "/home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0/CMakeLists.txt;33;add_test;/home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0/CMakeLists.txt;0;")
add_test(bin_test_CMT6 "/usr/bin/cmake" "-DINSTANCE=CMT6" "-DCOST=555.43" "-DROUND=0" "-P" "/home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0/Test/TestExecutable.cmake")
set_tests_properties(bin_test_CMT6 PROPERTIES  _BACKTRACE_TRIPLES "/home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0/CMakeLists.txt;40;add_test;/home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0/CMakeLists.txt;0;")
add_test(bin_test_CMT7 "/usr/bin/cmake" "-DINSTANCE=CMT7" "-DCOST=909.675" "-DROUND=0" "-P" "/home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0/Test/TestExecutable.cmake")
set_tests_properties(bin_test_CMT7 PROPERTIES  _BACKTRACE_TRIPLES "/home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0/CMakeLists.txt;45;add_test;/home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0/CMakeLists.txt;0;")
add_test(lib_test_c "/home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0/build/Test/Test-c/lib_test_c")
set_tests_properties(lib_test_c PROPERTIES  _BACKTRACE_TRIPLES "/home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0/CMakeLists.txt;57;add_test;/home/xunj/GradWork/Omni-VRP-main/POMO/CVRP/hgs/HGS-CVRP-2.0.0/CMakeLists.txt;0;")
subdirs("Test/Test-c")

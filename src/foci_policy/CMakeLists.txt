cmake_minimum_required(VERSION 3.8)
project(foci_policy)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/foci_node.py
  motion_planning/panda_control_velocity.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY motion_planning/
  DESTINATION lib/${PROJECT_NAME}/motion_planning
  USE_SOURCE_PERMISSIONS
)

# Install launch files
install(DIRECTORY
  launch config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

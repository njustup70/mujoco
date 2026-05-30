cmake_minimum_required(VERSION 3.8)
project(mujoco_ros2_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# Install Python scripts
set(SCRIPT_FILES
  scripts/mujoco_node.py
  scripts/odom_noise_node.py
  scripts/swerve_solver.py
  scripts/teleop_joy_node.py
  ../mpc_pkg/control_node.py
  ../mpc_pkg/decorder.py
  ../mpc_pkg/forcempc.py
  ../mpc_pkg/foxgloveTools.py
  ../mpc_pkg/linear.py
  ../mpc_pkg/mpc.py
  ../mpc_pkg/mpc_acados.py
  ../mpc_pkg/state_observer.py
)
install(
  PROGRAMS ${SCRIPT_FILES}
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch, model and config directories
install(
  DIRECTORY launch model config
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../models
  DESTINATION .
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

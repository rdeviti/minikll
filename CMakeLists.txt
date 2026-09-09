cmake_minimum_required(VERSION 3.14)

project(minikll VERSION 0.1.0 LANGUAGES CXX)

add_library(minikll INTERFACE)
add_library(minikll::minikll ALIAS minikll)

include(GNUInstallDirs)

target_compile_features(minikll INTERFACE cxx_std_17)
target_include_directories(
  minikll
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

option(MINIKLL_BUILD_TESTS "Build minikll tests" ON)
option(MINIKLL_BUILD_PYTHON "Build optional pybind11 Python bindings" OFF)
option(MINIKLL_BUILD_EXAMPLES "Build optional examples" OFF)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND MINIKLL_BUILD_TESTS)
  enable_testing()

  add_executable(minikll_tests tests/test_minikll.cpp)
  target_link_libraries(minikll_tests PRIVATE minikll::minikll)

  add_test(NAME minikll_tests COMMAND minikll_tests)
endif()

if(MINIKLL_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    RESULT_VARIABLE MINIKLL_PYBIND11_CMAKEDIR_RESULT
    OUTPUT_VARIABLE MINIKLL_PYBIND11_CMAKEDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(MINIKLL_PYBIND11_CMAKEDIR_RESULT EQUAL 0 AND EXISTS "${MINIKLL_PYBIND11_CMAKEDIR}")
    list(APPEND CMAKE_PREFIX_PATH "${MINIKLL_PYBIND11_CMAKEDIR}")
  endif()

  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(minikll_python bindings/python/minikll_pybind.cpp)
  target_link_libraries(minikll_python PRIVATE minikll::minikll)
  set_target_properties(minikll_python PROPERTIES OUTPUT_NAME minikll)

  set(
    MINIKLL_PYTHON_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH
    "Install directory for the minikll Python extension, relative to prefix unless absolute"
  )
  install(TARGETS minikll_python LIBRARY DESTINATION "${MINIKLL_PYTHON_INSTALL_DIR}")
endif()

if(MINIKLL_BUILD_EXAMPLES)
  find_package(HDF5 COMPONENTS CXX REQUIRED)

  add_executable(
    ann_benchmarks_distances
    examples/ann_benchmarks_distances.cpp
  )
  target_include_directories(ann_benchmarks_distances PRIVATE ${HDF5_INCLUDE_DIRS})
  target_compile_definitions(ann_benchmarks_distances PRIVATE ${HDF5_DEFINITIONS})
  target_link_libraries(
    ann_benchmarks_distances
    PRIVATE
      minikll::minikll
      ${HDF5_CXX_LIBRARIES}
  )

  install(TARGETS ann_benchmarks_distances RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE README.md CONTRIBUTING.md CHANGELOG.md docs/tutorial_ann_benchmarks.md DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(
  TARGETS minikll
  EXPORT minikllTargets
)
install(
  EXPORT minikllTargets
  NAMESPACE minikll::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/minikll
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/minikllConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/minikllConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/minikll
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/minikllConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
  ARCH_INDEPENDENT
)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/minikllConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/minikllConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/minikll
)

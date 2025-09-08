cmake_minimum_required(VERSION 3.12)

project(magpy_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 REQUIRED)

# Find nlohmann/json (for comparison tests)
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    message(STATUS "nlohmann/json found")
else()
    message(STATUS "nlohmann/json not found - will use FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Find ROOT (optional for ROOT file reading)
find_package(ROOT QUIET)
if(ROOT_FOUND)
    message(STATUS "ROOT found: ${ROOT_VERSION}")
    include(${ROOT_USE_FILE})
    add_definitions(-DHAS_ROOT)
else()
    message(WARNING "ROOT not found - ROOT file reading will be disabled")
endif()

# Add compiler flags for optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DEIGEN_NO_DEBUG")

# Include directories
include_directories(include)

# Add executable for tests
add_executable(test_magpy
    src/oscillator.cpp
    src/mc_event.cpp
    src/spline_handler.cpp
    src/bin_handler.cpp
    tests/test_oscillator.cpp
    src/sample_model.cpp
    src/spline_syst_model.cpp
    tests/test_runner.cpp
)

target_link_libraries(test_magpy Eigen3::Eigen)

# Add a simple test executable
add_executable(simple_test
    src/oscillator.cpp
    src/mc_event.cpp
    src/spline_handler.cpp
    src/bin_handler.cpp
    src/sample_model.cpp
    src/spline_syst_model.cpp
    simple_test.cpp
)

target_link_libraries(simple_test Eigen3::Eigen)

# Add benchmark executable
add_executable(benchmark_magpy
    src/oscillator.cpp
    src/mc_event.cpp
    src/spline_handler.cpp
    src/bin_handler.cpp
    src/sample_model.cpp
    src/spline_syst_model.cpp
    benchmark.cpp
)

target_link_libraries(benchmark_magpy Eigen3::Eigen)

# Add benchmark with data collection
add_executable(benchmark_with_data
    src/oscillator.cpp
    src/mc_event.cpp
    src/spline_handler.cpp
    src/bin_handler.cpp
    src/sample_model.cpp
    src/spline_syst_model.cpp
    benchmark_with_data.cpp
)

target_link_libraries(benchmark_with_data Eigen3::Eigen)

# Add ROOT file reading test (if ROOT is available)
if(ROOT_FOUND)
    add_executable(test_root_reading
        src/oscillator.cpp
        src/mc_event.cpp
        src/spline_handler.cpp
        src/bin_handler.cpp
        src/sample_model.cpp
        src/spline_syst_model.cpp
        src/mc_file.cpp
        test_root_reading.cpp
    )
    
    target_link_libraries(test_root_reading Eigen3::Eigen ${ROOT_LIBRARIES})
    target_include_directories(test_root_reading PRIVATE ${ROOT_INCLUDE_DIRS})
    
    message(STATUS "ROOT file reading test will be built")
else()
    message(STATUS "ROOT file reading test will be skipped (ROOT not found)")
endif()

# Add MC file reading test (CSV-based, no ROOT required)
add_executable(test_mc_reading
    src/oscillator.cpp
    src/mc_event.cpp
    src/spline_handler.cpp
    src/bin_handler.cpp
    src/sample_model.cpp
    src/spline_syst_model.cpp
    src/mc_file.cpp
    test_mc_reading.cpp
)

if(ROOT_FOUND)
    target_link_libraries(test_mc_reading Eigen3::Eigen ${ROOT_LIBRARIES})
    target_include_directories(test_mc_reading PRIVATE ${ROOT_INCLUDE_DIRS})
else()
    target_link_libraries(test_mc_reading Eigen3::Eigen)
endif()

# Enable testing
enable_testing()
add_test(NAME magpy_tests COMMAND test_magpy)

# Add comparison test executables
add_executable(test_oscillator_comparison
    src/oscillator.cpp
    src/mc_event.cpp
    src/spline_handler.cpp
    src/bin_handler.cpp
    src/sample_model.cpp
    src/spline_syst_model.cpp
    test_oscillator_comparison.cpp
)

target_link_libraries(test_oscillator_comparison Eigen3::Eigen nlohmann_json::nlohmann_json)

add_executable(test_mc_comparison
    src/oscillator.cpp
    src/mc_event.cpp
    src/spline_handler.cpp
    src/bin_handler.cpp
    src/sample_model.cpp
    src/spline_syst_model.cpp
    src/mc_file.cpp
    test_mc_comparison.cpp
)

if(ROOT_FOUND)
    target_link_libraries(test_mc_comparison Eigen3::Eigen ${ROOT_LIBRARIES} nlohmann_json::nlohmann_json)
    target_include_directories(test_mc_comparison PRIVATE ${ROOT_INCLUDE_DIRS})
else()
    target_link_libraries(test_mc_comparison Eigen3::Eigen nlohmann_json::nlohmann_json)
endif()

add_executable(test_spline_comparison
    src/oscillator.cpp
    src/mc_event.cpp
    src/spline_handler.cpp
    src/bin_handler.cpp
    src/sample_model.cpp
    src/spline_syst_model.cpp
    src/spline_file.cpp
    test_spline_comparison.cpp
)

if(ROOT_FOUND)
    target_link_libraries(test_spline_comparison Eigen3::Eigen ${ROOT_LIBRARIES} nlohmann_json::nlohmann_json)
    target_include_directories(test_spline_comparison PRIVATE ${ROOT_INCLUDE_DIRS})
    add_executable(test_spline_comparison_root
        src/oscillator.cpp
        src/mc_event.cpp
        src/spline_handler.cpp
        src/bin_handler.cpp
        src/sample_model.cpp
        src/spline_syst_model.cpp
        src/spline_file.cpp
        test_spline_comparison.cpp
    )
    target_link_libraries(test_spline_comparison_root Eigen3::Eigen ${ROOT_LIBRARIES} nlohmann_json::nlohmann_json)
    target_include_directories(test_spline_comparison_root PRIVATE ${ROOT_INCLUDE_DIRS})
else()
    target_link_libraries(test_spline_comparison Eigen3::Eigen nlohmann_json::nlohmann_json)
endif()

add_executable(performance_benchmark
    src/oscillator.cpp
    src/mc_event.cpp
    src/spline_handler.cpp
    src/bin_handler.cpp
    src/sample_model.cpp
    src/spline_syst_model.cpp
    src/mc_file.cpp
    performance_benchmark.cpp
)

if(ROOT_FOUND)
    target_link_libraries(performance_benchmark Eigen3::Eigen ${ROOT_LIBRARIES} nlohmann_json::nlohmann_json)
    target_include_directories(performance_benchmark PRIVATE ${ROOT_INCLUDE_DIRS})
else()
    target_link_libraries(performance_benchmark Eigen3::Eigen nlohmann_json::nlohmann_json)
endif()

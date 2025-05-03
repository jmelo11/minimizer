cmake_minimum_required(VERSION 3.10.0)
project(engine VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

# add the include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# add the libs
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/eigen)


# add the source directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
# add the lib directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

# add the include directory

file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "include/*.hpp")

add_executable(solver ${SOURCES} ${HEADERS})


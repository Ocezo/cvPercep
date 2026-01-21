cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project( cvPercep )

find_package( OpenCV REQUIRED )

add_executable( cvPercep cvPercep.cpp )

target_include_directories( cvPercep PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( cvPercep ${OpenCV_LIBS} )

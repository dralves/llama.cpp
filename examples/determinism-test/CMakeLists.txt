set(TARGET llama-determinism-test)
add_executable(${TARGET} determinism-test.cpp)

include_directories(deps/)

target_link_libraries(${TARGET} PRIVATE
    llama
    common
    ${CMAKE_THREAD_LIBS_INIT}
)

add_library(sha256-dt OBJECT deps/sha256/sha256.c deps/sha256/sha256.h)
target_link_libraries(${TARGET} PRIVATE sha256-dt)
target_compile_options(sha256-dt PRIVATE -w)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

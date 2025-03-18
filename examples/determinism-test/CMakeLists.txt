set(TARGET llama-determinism-test)
add_executable(${TARGET} determinism-test.cpp)
target_link_libraries(${TARGET} PRIVATE
    llama
    common
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

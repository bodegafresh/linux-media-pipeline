add_executable(lmp_smoke_tests
  smoke_tests.cpp
)
target_link_libraries(lmp_smoke_tests PRIVATE lmp::core)
lmp_apply_compiler_options(lmp_smoke_tests)

add_test(NAME lmp_smoke_tests COMMAND lmp_smoke_tests)

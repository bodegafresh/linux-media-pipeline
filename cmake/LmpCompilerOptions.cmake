function(lmp_apply_compiler_options target_name)
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4)
    if(LMP_ENABLE_WARNINGS_AS_ERRORS)
      target_compile_options(${target_name} PRIVATE /WX)
    endif()
    return()
  endif()

  target_compile_options(${target_name}
    PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wsign-conversion
      -Wshadow
      -Wold-style-cast
      -Wnon-virtual-dtor
  )

  if(LMP_ENABLE_WARNINGS_AS_ERRORS)
    target_compile_options(${target_name} PRIVATE -Werror)
  endif()
endfunction()

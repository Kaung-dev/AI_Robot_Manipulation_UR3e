file(REMOVE_RECURSE
  "libModbus.a"
  "libModbus.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/Modbus.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()

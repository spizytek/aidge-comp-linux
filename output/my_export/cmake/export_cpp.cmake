add_library(aidge_export_cpp INTERFACE)

# Propagated to the executable through the INTERFACE link.
option(AIDGE_CMP    "Compare feature maps against Aidge reference" OFF)
option(SAVE_OUTPUTS "Dump feature maps to files"                   OFF)

target_compile_definitions(aidge_export_cpp INTERFACE
    $<$<BOOL:${AIDGE_CMP}>:AIDGE_CMP>
    $<$<BOOL:${SAVE_OUTPUTS}>:SAVE_OUTPUTS>
)

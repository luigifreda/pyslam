# Creates C resources file from files in given directory
# Based on http://stackoverflow.com/a/27206982
function(embed_binary_files file_glob output)
    # Collect input files
    file(GLOB bins ${file_glob})
    # Stop when output file is newer than all binary files
    set(output_newer_than_bins 1)
    foreach(bin ${bins})
        if(bin IS_NEWER_THAN output)
            set(output_newer_than_bins 0)
            break()
        endif()
    endforeach()
    if(output_newer_than_bins)
        return()
    endif()
    # Create empty output file
    file(WRITE ${output} "")
    # Iterate through input files
    foreach(bin ${bins})
        # Get short filename
        string(REGEX MATCH "([^/]+)$" filename ${bin})
        # Replace filename spaces & extension separator for C compatibility
        string(REGEX REPLACE "\\.| " "_" filename ${filename})
        # Read hex data from file
        file(READ ${bin} filedata HEX)
        # Convert hex data for C compatibility
        string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," filedata ${filedata})
        # Append data to output file
        file(APPEND ${output} "extern const unsigned char ${filename}[] = {${filedata}};\nextern const unsigned ${filename}_size = sizeof(${filename});\n")
    endforeach()
endfunction()

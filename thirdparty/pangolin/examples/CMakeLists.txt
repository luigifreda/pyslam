# All examples depend on Pangolin GUI
if(BUILD_PANGOLIN_GUI)
    add_subdirectory(HelloPangolin)
    add_subdirectory(HelloPangolinOffscreen)
    add_subdirectory(HelloPangolinThreads)
    add_subdirectory(SimpleMultiDisplay)
    add_subdirectory(SimpleDisplayImage)
    add_subdirectory(SimpleScene)

    if(NOT HAVE_GLES OR HAVE_GLES_2)
        add_subdirectory(SimplePlot)
    endif()

    ## These samples require Pangolin Var support
    if(BUILD_PANGOLIN_VARS)
      add_subdirectory(SimpleDisplay)

      ## Video Samples require Pangolin Video support
      if(BUILD_PANGOLIN_VIDEO)
        add_subdirectory(SimpleVideo)
        add_subdirectory(SimpleRecord)
        add_subdirectory(SharedMemoryCamera)
      endif()

#      # This sample fails on many platforms, so exclude it for now,
#      # until we can create a better cmake test for support.
#      find_package(CUDA QUIET)
#      if( CUDA_FOUND )
#        add_subdirectory(VBODisplay)
#      endif()

    endif()
endif()

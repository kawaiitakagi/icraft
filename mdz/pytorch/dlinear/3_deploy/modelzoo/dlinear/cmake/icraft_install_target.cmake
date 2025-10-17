function(icraft_install_declare TARGET_NAME)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    
    string(TOLOWER ${PROJECT_NAME} LOWER_PROJECT_NAME)
    set(TARGETS_EXPORT_NAME             "${LOWER_PROJECT_NAME}-targets")

    install(TARGETS ${TARGET_NAME}
	    EXPORT ${TARGETS_EXPORT_NAME}
	    LIBRARY DESTINATION lib
	    ARCHIVE DESTINATION lib
	    RUNTIME DESTINATION bin
    )

    if(ARGN)
        if(NOT ARGN MATCHES "NULL")
            install(DIRECTORY ${ARGN}
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
        endif()
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include)
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    endif()

    if(MSVC)
        install(FILES $<TARGET_PDB_FILE:${TARGET_NAME}> DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
    endif()

endfunction()

function(icraft_install_apply VERSION_COMPAT)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(CMAKE_FILE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake)

    string(TOLOWER ${PROJECT_NAME} LOWER_PROJECT_NAME)
    set(TARGETS_EXPORT_NAME             "${LOWER_PROJECT_NAME}-targets")
    set(CONFIG_FILE_NAME                "${LOWER_PROJECT_NAME}-config")
    set(CMAKE_CONFIG_IN_FILE            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CONFIG_FILE_NAME}.cmake.in")
    set(CMAKE_CONFIG_FILE_BASE          "${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}")
    set(CMAKE_CONFIG_VERSION_FILE       "${CMAKE_CONFIG_FILE_BASE}-version.cmake")
    set(CMAKE_CONFIG_FILE               "${CMAKE_CONFIG_FILE_BASE}.cmake")

    configure_package_config_file(
        ${CMAKE_CONFIG_IN_FILE}
        ${CMAKE_CONFIG_FILE}
        INSTALL_DESTINATION ${CMAKE_FILE_DIR}
    )

    write_basic_package_version_file(
        ${CMAKE_CONFIG_VERSION_FILE}
        VERSION ${PACKAGE_VERSION}
        COMPATIBILITY ${VERSION_COMPAT} #ExactVersion # AnyNewerVersion
    )

    install(EXPORT ${TARGETS_EXPORT_NAME}
        DESTINATION ${CMAKE_FILE_DIR}
    )

    install(FILES  ${CMAKE_CONFIG_FILE} ${CMAKE_CONFIG_VERSION_FILE}
            DESTINATION ${CMAKE_FILE_DIR}
    )

endfunction()

function(icraft_install_module TARGET_NAME)
    set(options NOPDB OPTIONAL)
    cmake_parse_arguments(OPTION "${options}" "" "" ${ARGN})

    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS ${TARGET_NAME}
	    EXPORT ${TARGETS_EXPORT_NAME}
	    LIBRARY DESTINATION lib
	    RUNTIME DESTINATION bin
    )
    if(MSVC AND NOT OPTION_NOPDB)
        install(FILES $<TARGET_PDB_FILE:${TARGET_NAME}> DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
    endif()

endfunction()

function(icraft_add_alias TARGET_NAME ALIAS_NAME)
    add_library(${ALIAS_NAME} ALIAS ${TARGET_NAME})
    set_property(TARGET ${TARGET_NAME} PROPERTY EXPORT_NAME ${ALIAS_NAME})
endfunction()
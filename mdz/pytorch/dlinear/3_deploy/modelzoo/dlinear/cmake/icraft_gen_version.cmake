function(icraft_gen_version TARGET_NAME NAMESPACE VERSION_INCLUDE_DIR)
    set(VERSION_INFO_RC_IN [=[
1 VERSIONINFO
FILEVERSION @GIT_MAJOR_VERSION@, @GIT_MINOR_VERSION@, @GIT_PATCH_VERSION@, @GIT_COMMITSINCETAG@
PRODUCTVERSION @GIT_MAJOR_VERSION@, @GIT_MINOR_VERSION@, @GIT_PATCH_VERSION@, @GIT_COMMITSINCETAG@
FILEFLAGSMASK 0x3fL
#ifdef _DEBUG
FILEFLAGS 0x1L
#else
FILEFLAGS 0x0L
#endif
FILEOS 0x40004L
FILETYPE 0x2L
FILESUBTYPE 0x0L
BEGIN
    BLOCK "StringFileInfo"
    BEGIN
        BLOCK "080404b0"
        BEGIN
            VALUE "CompanyName", "Shanghai Fudan Microelectronics Group Co., Ltd."
            VALUE "FileDescription", "Icraft Apllication Components"
            VALUE "FileVersion", "@GIT_MAINVERSION@"
            VALUE "LegalCopyright", "Copyright FMSH(C) @BUILD_YEAR@"
            VALUE "ProductName", "Icraft Intelligent Computing Platform"
            VALUE "ProductVersion", "@GIT_FULLVERSION@"
        END
    END
    BLOCK "VarFileInfo"
    BEGIN
        VALUE "Translation", 0x804, 1200
    END
END
    ]=])

    set(GIT_VERSION_H_IN [=[
#pragma once

#include <string>

namespace @NAMESPACE@ {
	constexpr uint64_t MAJOR_VERSION_NUM = @GIT_MAJOR_VERSION@\\;
	constexpr uint64_t MINOR_VERSION_NUM = @GIT_MINOR_VERSION@\\;
	constexpr uint64_t PATCH_VERSION_NUM = @GIT_PATCH_VERSION@\\;
	constexpr uint64_t COMMITSINCETAG = @GIT_COMMITSINCETAG@\\;

	constexpr std::string_view COMMITID = "@GIT_COMMITID@"\\;
	constexpr std::string_view MODIFIED = "@GIT_MODIFIED@"\\;
	constexpr std::string_view BUILDTIME = "@BUILD_TIME@"\\;

	constexpr std::string_view MAIN_VERSION = "@GIT_MAINVERSION@"\\;
	constexpr std::string_view FULL_VERSION = "@GIT_FULLVERSION@"\\;

    static constexpr std::string_view __version_string__ = "full_version: @GIT_FULLVERSION@"\\;
}
    ]=])

    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --tags --always --long
            OUTPUT_VARIABLE GIT_DESCRIBE
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        )
        execute_process(
            COMMAND ${GIT_EXECUTABLE} status --porcelain=v1
            OUTPUT_VARIABLE GIT_STATUS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        )
    endif()
    if(GIT_DESCRIBE)
        string(REGEX MATCH "v([0-9]+)\\.([0-9]+)\\.([0-9]+)-([0-9]+)-g([0-9a-f]+)" _ ${GIT_DESCRIBE})
        set(GIT_MAJOR_VERSION ${CMAKE_MATCH_1})
        set(GIT_MINOR_VERSION ${CMAKE_MATCH_2})
        set(GIT_PATCH_VERSION ${CMAKE_MATCH_3})
        set(GIT_COMMITSINCETAG ${CMAKE_MATCH_4})
        set(GIT_COMMITID ${CMAKE_MATCH_5})
    endif()

    set(GIT_MODIFIED "")
    if(GIT_STATUS)
        set(GIT_MODIFIED "+m")
    endif()
    string(TIMESTAMP BUILD_TIME %y%m%d%H%M)

    set(GIT_MAINVERSION "${GIT_MAJOR_VERSION}.${GIT_MINOR_VERSION}.${GIT_PATCH_VERSION}.${GIT_COMMITSINCETAG}")
    set(GIT_FULLVERSION "${GIT_MAINVERSION}-${GIT_COMMITID}${GIT_MODIFIED}(${BUILD_TIME})")

    set(PACKAGE_VERSION ${GIT_MAINVERSION} PARENT_SCOPE)
    message(STATUS "Generating version of ${TARGET_NAME} : ${GIT_FULLVERSION}")

    set(VERSION_FILES_DIR ${CMAKE_BINARY_DIR}/_gitver)
    string(CONFIGURE ${GIT_VERSION_H_IN} GIT_VERSION_H @ONLY)
    file(WRITE ${VERSION_FILES_DIR}/${VERSION_INCLUDE_DIR}/_git_version.h ${GIT_VERSION_H})

    get_target_property(TARGET_TYPE ${TARGET_NAME} TYPE)
    if (TARGET_TYPE STREQUAL "INTERFACE_LIBRARY")
        target_include_directories(${TARGET_NAME}  INTERFACE
            $<BUILD_INTERFACE:${VERSION_FILES_DIR}>
        )
    else()
        target_include_directories(${TARGET_NAME}  PUBLIC
            $<BUILD_INTERFACE:${VERSION_FILES_DIR}>
        )

        string(TIMESTAMP BUILD_YEAR %Y)
        string(CONFIGURE ${VERSION_INFO_RC_IN} VERSION_INFO_RC @ONLY)
        file(WRITE ${VERSION_FILES_DIR}/${TARGET_NAME}_version_info.rc ${VERSION_INFO_RC})

        target_sources(${TARGET_NAME} PRIVATE
            ${VERSION_FILES_DIR}/${TARGET_NAME}_version_info.rc
        )
    endif()

    if(NOT VERSION_INCLUDE_DIR MATCHES "NULL")
        include(GNUInstallDirs)
        install(DIRECTORY ${VERSION_FILES_DIR}/${VERSION_INCLUDE_DIR}/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${VERSION_INCLUDE_DIR}/)
    endif()
endfunction()
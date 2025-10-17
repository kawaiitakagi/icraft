set(CONAN_MINIMUM_VERSION 2.0.5)

function(dep_copy PKG_NAME)
    # copy deps to ${CMAKE_BINARY_DIR} manually
    string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE_UPPER)
    set(DEP_BIN_DIR "${${PKG_NAME}_PACKAGE_FOLDER_${BUILD_TYPE_UPPER}}/bin")
    if(EXISTS ${DEP_BIN_DIR})
        file(GLOB ALL_DEPS "${DEP_BIN_DIR}/*")
        foreach(dep ${ALL_DEPS})
            message(STATUS "Deps to be copied: ${dep}")
        endforeach()
        file(COPY ${ALL_DEPS} DESTINATION ${CMAKE_BINARY_DIR} FILES_MATCHING PATTERN "*")
    endif()
endfunction()

function(detect_os OS)
    # it could be cross compilation
    message(STATUS "CMake-Conan: cmake_system_name=${CMAKE_SYSTEM_NAME}")
    if(CMAKE_SYSTEM_NAME AND NOT CMAKE_SYSTEM_NAME STREQUAL "Generic")
        if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
            set(${OS} Macos PARENT_SCOPE)
        elseif(${CMAKE_SYSTEM_NAME} STREQUAL "QNX")
            set(${OS} Neutrino PARENT_SCOPE)
        else()
            set(${OS} ${CMAKE_SYSTEM_NAME} PARENT_SCOPE)
        endif()
    endif()
endfunction()

function(detect_arch ARCH)
    # it could be cross compilation
    message(STATUS "CMake-Conan: cmake_system_processor=${CMAKE_SYSTEM_PROCESSOR}")
    if(CMAKE_SYSTEM_NAME AND NOT CMAKE_SYSTEM_NAME STREQUAL "Generic")
        if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv8" 
        OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64"
        OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64"
        )
            set(${ARCH} armv8 PARENT_SCOPE)
        elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64" 
            OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64"
        )
            set(${ARCH} x86_64 PARENT_SCOPE)
        else()
            message(FATAL_ERROR "cmake_system_processor ${CMAKE_SYSTEM_PROCESSOR} is NOT supported")
        endif()
    endif()
endfunction()


function(detect_cxx_standard CXX_STANDARD)
    set(${CXX_STANDARD} ${CMAKE_CXX_STANDARD} PARENT_SCOPE)
    if (CMAKE_CXX_EXTENSIONS)
        set(${CXX_STANDARD} "gnu${CMAKE_CXX_STANDARD}" PARENT_SCOPE)
    endif()
endfunction()


function(detect_compiler COMPILER COMPILER_VERSION)
    if(DEFINED CMAKE_CXX_COMPILER_ID)
        set(_COMPILER ${CMAKE_CXX_COMPILER_ID})
        set(_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})
    else()
        if(NOT DEFINED CMAKE_C_COMPILER_ID)
            message(FATAL_ERROR "C or C++ compiler not defined")
        endif()
        set(_COMPILER ${CMAKE_C_COMPILER_ID})
        set(_COMPILER_VERSION ${CMAKE_C_COMPILER_VERSION})
    endif()

    message(STATUS "CMake-Conan: CMake compiler=${_COMPILER}")
    message(STATUS "CMake-Conan: CMake compiler version=${_COMPILER_VERSION}")

    if(_COMPILER MATCHES MSVC)
        set(_COMPILER "msvc")
        string(SUBSTRING ${MSVC_VERSION} 0 3 _COMPILER_VERSION)
    elseif(_COMPILER MATCHES AppleClang)
        set(_COMPILER "apple-clang")
        string(REPLACE "." ";" VERSION_LIST ${CMAKE_CXX_COMPILER_VERSION})
        list(GET VERSION_LIST 0 _COMPILER_VERSION)
    elseif(_COMPILER MATCHES Clang)
        set(_COMPILER "clang")
        string(REPLACE "." ";" VERSION_LIST ${CMAKE_CXX_COMPILER_VERSION})
        list(GET VERSION_LIST 0 _COMPILER_VERSION)
    elseif(_COMPILER MATCHES GNU)
        set(_COMPILER "gcc")
        string(REPLACE "." ";" VERSION_LIST ${CMAKE_CXX_COMPILER_VERSION})
        list(GET VERSION_LIST 0 _COMPILER_VERSION)
    endif()

    message(STATUS "CMake-Conan: [settings] compiler=${_COMPILER}")
    message(STATUS "CMake-Conan: [settings] compiler.version=${_COMPILER_VERSION}")

    set(${COMPILER} ${_COMPILER} PARENT_SCOPE)
    set(${COMPILER_VERSION} ${_COMPILER_VERSION} PARENT_SCOPE)
endfunction()

function(detect_build_type BUILD_TYPE)
    if(NOT CMAKE_CONFIGURATION_TYPES)
        # Only set when we know we are in a single-configuration generator
        # Note: we may want to fail early if `CMAKE_BUILD_TYPE` is not defined
        set(${BUILD_TYPE} ${CMAKE_BUILD_TYPE} PARENT_SCOPE)
    endif()
endfunction()


function(detect_host_profile output_file)
    detect_os(MYOS)
    detect_arch(MYARCH)
    detect_compiler(MYCOMPILER MYCOMPILER_VERSION)
    detect_cxx_standard(MYCXX_STANDARD)
    detect_build_type(MYBUILD_TYPE)

    set(PROFILE "")
    string(APPEND PROFILE "include(default)\n")
    string(APPEND PROFILE "[settings]\n")
    if(MYOS)
        string(APPEND PROFILE os=${MYOS} "\n")
    endif()
    if(MYARCH)
        string(APPEND PROFILE arch=${MYARCH} "\n")
    endif()
    if(MYCOMPILER)
        string(APPEND PROFILE compiler=${MYCOMPILER} "\n")
    endif()
    if(MYCOMPILER_VERSION)
        string(APPEND PROFILE compiler.version=${MYCOMPILER_VERSION} "\n")
    endif()
    if(MYCXX_STANDARD)
        string(APPEND PROFILE compiler.cppstd=${MYCXX_STANDARD} "\n")
    endif()
    if(MYBUILD_TYPE)
        string(APPEND PROFILE "build_type=${MYBUILD_TYPE}\n")
    endif()

    if(NOT DEFINED output_file)
        set(_FN "${CMAKE_BINARY_DIR}/profile")
    else()
        set(_FN ${output_file})
    endif()

    string(APPEND PROFILE "[conf]\n")
    string(APPEND PROFILE "tools.cmake.cmaketoolchain:generator=${CMAKE_GENERATOR}\n")

    message(STATUS "CMake-Conan: Creating profile ${_FN}")
    file(WRITE ${_FN} ${PROFILE})
    message(STATUS "CMake-Conan: Profile: \n${PROFILE}")
endfunction()


function(conan_profile_detect_default)
    message(STATUS "CMake-Conan: Checking if a default profile exists")
    execute_process(COMMAND ${CONAN_COMMAND} profile path default
                    RESULT_VARIABLE return_code
                    OUTPUT_VARIABLE conan_stdout
                    ERROR_VARIABLE conan_stderr
                    ECHO_ERROR_VARIABLE    # show the text output regardless
                    ECHO_OUTPUT_VARIABLE
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    if(NOT ${return_code} EQUAL "0")
        message(STATUS "CMake-Conan: The default profile doesn't exist, detecting it.")
        execute_process(COMMAND ${CONAN_COMMAND} profile detect
            RESULT_VARIABLE return_code
            OUTPUT_VARIABLE conan_stdout
            ERROR_VARIABLE conan_stderr
            ECHO_ERROR_VARIABLE    # show the text output regardless
            ECHO_OUTPUT_VARIABLE
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()
endfunction()

# !! conan_install modified to feat FetchContent dependencies -- author LZN
function(conan_install)
    # 获取fetchcontent路径下的文件名称
    file(GLOB dep_dirs ${FETCHCONTENT_BASE_DIR}/*)
    # 获取ROOT_INSTALLED变量，为TRUE说明已经安装过顶层CMAKE下的conanfiles
    get_property(ROOT_INSTALLED GLOBAL PROPERTY ROOT_INSTALLED)
    # 如果ROOT_INTSALLED为FALSE，把顶层conanfile路径添加到安装路径集，设置ROOT_INSTALLED为TRUE，避免下次再次安装
    if(NOT ROOT_INSTALLED)
        set_property(GLOBAL PROPERTY ROOT_INSTALLED TRUE)
        list(APPEND dep_dirs ${CMAKE_SOURCE_DIR})
        set(CMAKE_INSTALL_DIRS ${CMAKE_SOURCE_DIR})
    else()
        set(CMAKE_INSTALL_DIRS "")
    endif()
    foreach(dep_dir ${dep_dirs})
        if(dep_dir MATCHES "-src")
            message(STATUS "check dep dir: ${dep_dir}")
            if(EXISTS ${dep_dir}/conanfile.txt)
                set(APPEND_INSTALL_DIRS FALSE)
                # debug: read conanfile.txt get package name to be installed
                file(READ ${dep_dir}/conanfile.txt FILE_CONTENTS)
                string(REGEX REPLACE "\n$" "" FILE_CONTENTS "${FILE_CONTENTS}")
                string(REGEX REPLACE "\n" ";" FILE_LINES "${FILE_CONTENTS}")
                # 遍历目标路径下的conanfile，如果有没被安装的包就设置APPEND_INSTALL_DIRS为TRUE
                foreach(line IN LISTS FILE_LINES)
                    if(line MATCHES "/")
                        string(REPLACE "/" ";" PKG_LINE_LIST "${line}")
                        list(GET PKG_LINE_LIST 0 PKG_NAME)
                        # debug: check if XXX_INSTALLED, if true, skip
                        get_property(PKG_INSTALLED GLOBAL PROPERTY ${PKG_NAME}_INSTALLED)
                        message(STATUS "pkg name: ${PKG_NAME}")
                        if(NOT PKG_INSTALLED)
                            set(APPEND_INSTALL_DIRS TRUE)
                            message(STATUS "setting ${PKG_NAME}_INSTALLED")
                            set_property(GLOBAL PROPERTY ${PKG_NAME}_INSTALLED TRUE)
                        endif()
                    endif()                    
                endforeach()
                # 如果APPEND_INSTALL_DIRS为TRUE，把当前路径添加到安装路径集
                if(APPEND_INSTALL_DIRS)
                    message(STATUS "Conan install list: ${dep_dir}")
                    # CMAKE_INSTALL_DIRS: 保存所有需要安装的conanfile.txt路径
                    list(APPEND CMAKE_INSTALL_DIRS ${dep_dir})
                else()
                    message(STATUS "skip dep dir ${dep_dir}")
                endif()
            endif()
        endif()
    endforeach()
    # 遍历安装路径集，安装conanfile中声明的conan包
    foreach(CMAKE_INSTALL_DIR ${CMAKE_INSTALL_DIRS})
        cmake_parse_arguments(ARGS CONAN_ARGS ${ARGN})
        set(CONAN_OUTPUT_FOLDER ${CMAKE_BINARY_DIR}/conan)
        # Invoke "conan install" with the provided arguments
        set(CONAN_ARGS ${CONAN_ARGS} -of=${CONAN_OUTPUT_FOLDER})

        message(STATUS "CMake-Conan: conan install ${CMAKE_INSTALL_DIR} ${CONAN_ARGS} ${ARGN}")
        execute_process(COMMAND ${CONAN_COMMAND} install ${CMAKE_INSTALL_DIR} ${CONAN_ARGS} ${ARGN} --format=json
                        RESULT_VARIABLE return_code
                        OUTPUT_VARIABLE conan_stdout
                        ERROR_VARIABLE conan_stderr
                        ECHO_ERROR_VARIABLE    # show the text output regardless
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        if(NOT "${return_code}" STREQUAL "0")
            message(FATAL_ERROR "Conan install failed='${return_code}'")
        else()
            # the files are generated in a folder that depends on the layout used, if
            # one is specified, but we don't know a priori where this is.
            # TODO: this can be made more robust if Conan can provide this in the json output
            string(JSON CONAN_GENERATORS_FOLDER GET ${conan_stdout} graph nodes 0 generators_folder)
            # message("conan stdout: ${conan_stdout}")
            message(STATUS "CMake-Conan: CONAN_GENERATORS_FOLDER=${CONAN_GENERATORS_FOLDER}")
            set_property(GLOBAL PROPERTY CONAN_GENERATORS_FOLDER "${CONAN_GENERATORS_FOLDER}")
            # reconfigure on conanfile changes
            string(JSON CONANFILE GET ${conan_stdout} graph nodes 0 label)
            message(STATUS "CMake-Conan: CONANFILE=${CMAKE_INSTALL_DIR}/${CONANFILE}")
            set_property(DIRECTORY ${CMAKE_SOURCE_DIR} APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_INSTALL_DIR}/${CONANFILE}")
        endif()
    endforeach()
endfunction()


function(conan_get_version conan_command conan_current_version)
    execute_process(
        COMMAND ${conan_command} --version
        OUTPUT_VARIABLE conan_output
        RESULT_VARIABLE conan_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(conan_result)
        message(FATAL_ERROR "CMake-Conan: Error when trying to run Conan")
    endif()

    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" conan_version ${conan_output})
    set(${conan_current_version} ${conan_version} PARENT_SCOPE)
endfunction()


function(conan_version_check)
    set(options )
    set(oneValueArgs MINIMUM CURRENT)
    set(multiValueArgs )
    cmake_parse_arguments(CONAN_VERSION_CHECK
        "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT CONAN_VERSION_CHECK_MINIMUM)
        message(FATAL_ERROR "CMake-Conan: Required parameter MINIMUM not set!")
    endif()
        if(NOT CONAN_VERSION_CHECK_CURRENT)
        message(FATAL_ERROR "CMake-Conan: Required parameter CURRENT not set!")
    endif()

    if(CONAN_VERSION_CHECK_CURRENT VERSION_LESS CONAN_VERSION_CHECK_MINIMUM)
        message(FATAL_ERROR "CMake-Conan: Conan version must be ${CONAN_VERSION_CHECK_MINIMUM} or later")
    endif()
endfunction()


macro(conan_provide_dependency package_name)
    if(${ARGV1} STREQUAL "Git")
        set(GIT_REQUIRED TRUE)
    else()
        set(GIT_REQUIRED FALSE)
    endif()
    if(NOT ${ARGV1}_INSTALLED AND NOT GIT_REQUIRED)
        find_program(CONAN_COMMAND "conan" REQUIRED)
        conan_get_version(${CONAN_COMMAND} CONAN_CURRENT_VERSION)
        conan_version_check(MINIMUM ${CONAN_MINIMUM_VERSION} CURRENT ${CONAN_CURRENT_VERSION})
        message(STATUS "CMake-Conan: first find_package() found. Installing dependencies with Conan")
        conan_profile_detect_default()
        detect_host_profile(${CMAKE_BINARY_DIR}/conan_host_profile)
        if(NOT CMAKE_CONFIGURATION_TYPES)
            message(STATUS "CMake-Conan: Installing single configuration ${CMAKE_BUILD_TYPE}")
            conan_install(-pr ${CMAKE_BINARY_DIR}/conan_host_profile --build=missing -g CMakeDeps)
        else()
            message(STATUS "CMake-Conan: Installing both Debug and Release")
            conan_install(-pr ${CMAKE_BINARY_DIR}/conan_host_profile -s build_type=Release --build=missing -g CMakeDeps)
            conan_install(-pr ${CMAKE_BINARY_DIR}/conan_host_profile -s build_type=Debug --build=missing -g CMakeDeps)
        endif()
    else()
        message(STATUS "CMake-Conan: find_package(${ARGV1}) found, 'conan install' already ran")
    endif()

    get_property(CONAN_GENERATORS_FOLDER GLOBAL PROPERTY CONAN_GENERATORS_FOLDER)
    list(FIND CMAKE_PREFIX_PATH "${CONAN_GENERATORS_FOLDER}" index)
    if(${index} EQUAL -1)
        list(PREPEND CMAKE_PREFIX_PATH "${CONAN_GENERATORS_FOLDER}")
    endif()
    find_package(${ARGN} BYPASS_PROVIDER)
endmacro()

cmake_language(
  SET_DEPENDENCY_PROVIDER conan_provide_dependency
  SUPPORTED_METHODS FIND_PACKAGE
)
# 
# enables testing with google test
# provides add_gtest(test) and add_gtest_release(test) commands
#

# check xml output
option(TestXMLOutput "Output tests to xml" OFF)

# custom function to add gtest with xml output
# arg0 - test (assume the source is ${test_name}.cpp
function(add_gtest test_name)
    if (TestXMLOutput)
        set (test_xml_output --gtest_output=xml:${test_name}.xml)
    endif(TestXMLOutput)

    set(source "test/${test_name}.cpp")
    set(gtest_src "test/gtest_main.cc;test/gtest-all.cc")
    #message(status ${ARGV1})

    add_executable(${test_name} ${source} ${gtest_src})
    target_link_libraries(${test_name} ${LINK_ALL})
    add_test(NAME ${test_name} COMMAND ${test_name} ${test_xml_output})
endfunction(add_gtest)


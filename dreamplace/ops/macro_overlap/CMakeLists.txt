set(OP_NAME macro_overlap)

# set(TARGET_NAME ${OP_NAME})


# install(TARGETS ${TARGET_NAME}_cpp DESTINATION dreamplace/ops/${OP_NAME})
file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
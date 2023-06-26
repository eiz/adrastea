load("@rules_python//python:defs.bzl", "py_binary")
load("@py_deps//:requirements.bzl", "requirement")

py_binary(
  name = "gen_tests",
  srcs = ["gen_tests.py"],
  deps = [
    requirement("numpy"),
    requirement("torch"),
  ]
)

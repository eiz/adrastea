load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_python",
    sha256 = "84aec9e21cc56fbc7f1335035a71c850d1b9b5cc6ff497306f84cced9a769841",
    strip_prefix = "rules_python-0.23.1",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.23.1/rules_python-0.23.1.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("@rules_python//python:repositories.bzl", "python_register_toolchains")

python_register_toolchains(
    name = "python3_11",
    python_version = "3.11",
)

load("@python3_11//:defs.bzl", "interpreter")

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "py_deps",
    requirements_lock = "//py:requirements.txt",
    python_interpreter_target = interpreter,
    extra_pip_args = ["--extra-index-url", "https://download.pytorch.org/whl/cpu"],
)

load("@py_deps//:requirements.bzl", "install_deps")

install_deps()

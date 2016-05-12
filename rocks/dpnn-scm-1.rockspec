package = "dpnn"
version = "scm-1"

source = {
   url = "git://github.com/Element-Research/dpnn",
   tag = "master"
}

description = {
   summary = "deep extensions to nn Modules and Criterions",
   detailed = [[sharedClone, type, outside, updateGradParameters, Serial, Inception, etc.]],
   homepage = "https://github.com/Element-Research/dpnn",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
   "nnx >= 0.1",
   "moses >= 1.3.1"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUAROCKS_PREFIX)" -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}

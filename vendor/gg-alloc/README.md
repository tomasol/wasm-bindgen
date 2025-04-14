# gg-alloc

A custom allocator that always returns pointers in the 2G-4G range.

That is, pointers that are valid `u32` but not valid `i32`. This is used to test the wasm-bindgen crate, which used to have problems when dealing with more than 2GB of memory in WebAssembly because of this.

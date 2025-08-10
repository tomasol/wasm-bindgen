use std::alloc::{GlobalAlloc, Layout};

pub struct GgAlloc<A> {
    alloc: A,
}

impl<A> GgAlloc<A> {
    pub const fn new(alloc: A) -> Self {
        Self { alloc }
    }
}

fn pointer_above_2g(ptr: *mut u8) -> bool {
    //(ptr as usize) > (isize::MAX as usize)
    (ptr as u32) > (i32::MAX as u32)
}

fn alloc_fully_below_2g(ptr: *mut u8, layout: Layout) -> bool {
    !pointer_above_2g(ptr) && !pointer_above_2g(unsafe { ptr.add(layout.size() - 1) })
}

unsafe impl<A> GlobalAlloc for GgAlloc<A>
where
    A: GlobalAlloc,
{
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let mut ret = self.alloc.alloc(layout);

        loop {
            if ret.is_null() {
                // Failed to allocate, return null pointer
                break;
            }

            if pointer_above_2g(ret) {
                // Allocated above 2G, return this pointer
                break;
            }

            // Here the allocation was successful but below 2G, so fill memory a bit
            // in chunks of 128 MB
            let mut size = 1 << 27;
            loop {
                let test_layout = Layout::from_size_align(size, 1).unwrap();
                let fill_ptr = self.alloc.alloc(test_layout);
                if !fill_ptr.is_null() && alloc_fully_below_2g(fill_ptr, test_layout) {
                    // leak allocation and continue
                } else {
                    // Free and try again with half size
                    if !fill_ptr.is_null() {
                        self.alloc.dealloc(fill_ptr, test_layout);
                    }
                    size /= 2;

                    if size < 1 {
                        // Exit fill loop if size is small
                        break;
                    }
                }
            }

            if !alloc_fully_below_2g(ret, layout) {
                // Free and allocate something with half size
                self.alloc.dealloc(ret, layout);
            }

            // fill memory a bit
            // in chunks of same size as the requested allocation
            loop {
                let test_layout = layout;
                let fill_ptr = self.alloc.alloc(test_layout);
                if !fill_ptr.is_null() && alloc_fully_below_2g(fill_ptr, test_layout) {
                    // leak allocation and continue
                } else {
                    // this allocation is either above 2GB or right at the 2GB edge, we are done
                    if pointer_above_2g(fill_ptr) {
                        // above 2GB, we can use it as the return pointer
                        return fill_ptr;
                    } else {
                        // right at the edge, leak this allocation and try again in next iteration
                        // of the outer loop
                        break;
                    }
                }
            }

            ret = self.alloc.alloc(layout);
        }

        ret
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.alloc.dealloc(ptr, layout);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::System;

    #[global_allocator]
    static A: GgAlloc<System> = GgAlloc::new(System);

    fn assert_above_2g(ptr: *mut u8) {
        assert!(pointer_above_2g(ptr), "{:p}", ptr,);
    }

    #[test]
    fn alloc_string() {
        let s = format!("allocating a string!");
        assert_above_2g(s.as_str() as *const _ as *mut _);
    }

    #[test]
    fn alloc_one_byte() {
        let s = format!("1");
        assert!(
            pointer_above_2g(s.as_str() as *const _ as *mut _),
            "{:p}",
            s.as_str()
        );
    }

    #[test]
    fn alloc_many_bytes() {
        let s = format!("1");
        assert_above_2g(s.as_str() as *const _ as *mut _);
        let s = format!("12");
        assert_above_2g(s.as_str() as *const _ as *mut _);
        let s = format!("123");
        assert_above_2g(s.as_str() as *const _ as *mut _);
        let s = format!("1234");
        assert_above_2g(s.as_str() as *const _ as *mut _);
        let s = format!("12345");
        assert_above_2g(s.as_str() as *const _ as *mut _);
        let s = format!("123456");
        assert_above_2g(s.as_str() as *const _ as *mut _);
        let s = format!("1234567");
        assert_above_2g(s.as_str() as *const _ as *mut _);
        let s = format!("12345678");
        assert_above_2g(s.as_str() as *const _ as *mut _);
        let s = format!("123456789");
        assert_above_2g(s.as_str() as *const _ as *mut _);
    }
}

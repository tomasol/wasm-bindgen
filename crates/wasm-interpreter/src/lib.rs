//! A tiny and incomplete Wasm interpreter
//!
//! This module contains a tiny and incomplete Wasm interpreter built on top of
//! `walrus`'s module structure. Each `Interpreter` contains some state
//! about the execution of a Wasm instance. The "incomplete" part here is
//! related to the fact that this is *only* used to execute the various
//! descriptor functions for wasm-bindgen.
//!
//! As a recap, the wasm-bindgen macro generate "descriptor functions" which
//! basically as a mapping of rustc's trait resolution in executable code. This
//! allows us to detect, after the macro is invoke, what trait selection did and
//! what types of functions look like. By executing descriptor functions they'll
//! each invoke a known import (with only one argument) some number of times,
//! which gives us a list of `u32` values to then decode.
//!
//! The interpreter here is only geared towards this one exact use case, so it's
//! quite small and likely not extra-efficient.

#![deny(missing_docs)]

use anyhow::{bail, ensure};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use walrus::ir::Instr;
use walrus::{ElementId, FunctionId, LocalId, Module, TableId};

/// A ready-to-go interpreter of a Wasm module.
///
/// An interpreter currently represents effectively cached state. It is reused
/// between calls to `interpret` and is precomputed from a `Module`. It houses
/// state like the Wasm stack, Wasm memory, etc.
#[derive(Default)]
pub struct Interpreter {
    // Function index of the `__wbindgen_describe` and
    // `__wbindgen_describe_closure` imported functions. We special case this
    // to know when the environment's imported function is called.
    describe_id: Option<FunctionId>,
    describe_closure_id: Option<FunctionId>,

    // Id of the function table
    functions: Option<TableId>,

    // A mapping of string names to the function index, filled with all exported
    // functions.
    name_map: HashMap<String, FunctionId>,

    // The current stack pointer (global 0) and Wasm memory (the stack). Only
    // used in a limited capacity.
    sp: i32,
    mem: Vec<i32>,
    scratch: Vec<i32>,

    // The descriptor which we're assembling, a list of `u32` entries. This is
    // very specific to wasm-bindgen and is the purpose for the existence of
    // this module.
    descriptor: Vec<u32>,

    // When invoking the `__wbindgen_describe_closure` imported function, this
    // stores the last table index argument, used for finding a different
    // descriptor.
    descriptor_table_idx: Option<u32>,
}

impl Interpreter {
    /// Creates a new interpreter from a provided `Module`, precomputing all
    /// information necessary to interpret further.
    ///
    /// Note that the `module` passed in to this function must be the same as
    /// the `module` passed to `interpret` below.
    pub fn new(module: &Module) -> Result<Interpreter, anyhow::Error> {
        let mut ret = Interpreter::default();

        // Give ourselves some memory and set the stack pointer
        // (the LLVM call stack, now the Wasm stack, global 0) to the top.
        ret.mem = vec![0; 1_000_000];
        ret.sp = ret.mem.len() as i32;

        // Figure out where the `__wbindgen_describe` imported function is, if
        // it exists. We'll special case calls to this function as our
        // interpretation should only invoke this function as an imported
        // function.
        for import in module.imports.iter() {
            let id = match import.kind {
                walrus::ImportKind::Function(id) => id,
                _ => continue,
            };
            if import.module != "__wbindgen_placeholder__" {
                continue;
            }
            if import.name == "__wbindgen_describe" {
                ret.describe_id = Some(id);
            } else if import.name == "__wbindgen_describe_closure" {
                ret.describe_closure_id = Some(id);
            }
        }

        // Build up the mapping of exported functions to function ids.
        for export in module.exports.iter() {
            let id = match export.item {
                walrus::ExportItem::Function(id) => id,
                _ => continue,
            };
            ret.name_map.insert(export.name.to_string(), id);
        }

        ret.functions = module.tables.main_function_table()?;

        Ok(ret)
    }

    /// Interprets the execution of the descriptor function `func`.
    ///
    /// This function will execute `func` in the `module` provided. Note that
    /// the `module` provided here must be the same as the one passed to `new`
    /// when this `Interpreter` was constructed.
    ///
    /// The `func` must be a wasm-bindgen descriptor function meaning that it
    /// doesn't do anything like use floats or i64. Instead all it should do is
    /// call other functions, sometimes some stack pointer manipulation, and
    /// then call the one imported `__wbindgen_describe` function. Anything else
    /// will cause this interpreter to panic.
    ///
    /// When the descriptor has finished running the assembled descriptor list
    /// is returned. The descriptor returned can then be re-parsed into an
    /// actual `Descriptor` in the cli-support crate.
    ///
    /// # Return value
    ///
    /// Returns `Some` if `func` was found in the `module` and `None` if it was
    /// not found in the `module`.
    pub fn interpret_descriptor(&mut self, id: FunctionId, module: &Module) -> Option<&[u32]> {
        self.descriptor.truncate(0);

        // We should have a blank Wasm and LLVM stack at both the start and end
        // of the call.
        assert_eq!(self.sp, self.mem.len() as i32);
        self.call(id, module, &[]);
        assert_eq!(self.sp, self.mem.len() as i32);
        Some(&self.descriptor)
    }

    /// Interprets a "closure descriptor", figuring out the signature of the
    /// closure that was intended.
    ///
    /// This function will take an `id` which is known to internally
    /// execute `__wbindgen_describe_closure` and interpret it. The
    /// `wasm-bindgen` crate controls all callers of this internal import. It
    /// will then take the index passed to `__wbindgen_describe_closure` and
    /// interpret it as a function pointer. This means it'll look up within the
    /// element section (function table) which index it points to. Upon finding
    /// the relevant entry it'll assume that function is a descriptor function,
    /// and then it will execute the descriptor function.
    ///
    /// The returned value is the return value of the descriptor function found.
    /// The `entry_removal_list` list is also then populated with an index of
    /// the entry in the elements section (and then the index within that
    /// section) of the function that needs to be snip'd out.
    pub fn interpret_closure_descriptor(
        &mut self,
        id: FunctionId,
        module: &Module,
        entry_removal_list: &mut HashMap<ElementId, BTreeSet<usize>>,
    ) -> Option<&[u32]> {
        // Call the `id` function. This is an internal `#[inline(never)]`
        // whose code is completely controlled by the `wasm-bindgen` crate, so
        // it should take some arguments (the number of arguments depends on the
        // optimization level) and return one (all of which we don't care about
        // here). What we're interested in is that while executing this function
        // it'll call `__wbindgen_describe_closure` with an argument that we
        // look for.
        assert!(self.descriptor_table_idx.is_none());

        let func = module.funcs.get(id);
        let params = module.types.get(func.ty()).params();
        assert!(
            params.iter().all(|p| *p == walrus::ValType::I32),
            "closure descriptors should only have i32 params"
        );
        let num_params = params.len();
        assert!(
            num_params <= 2,
            "closure descriptors have 2 parameters, but might lose some parameters due to LTO"
        );

        let args = vec![0; num_params];
        self.call(id, module, &args);
        let descriptor_table_idx = self
            .descriptor_table_idx
            .take()
            .expect("descriptor function should return index");

        // After we've got the table index of the descriptor function we're
        // interested go take a look in the function table to find what the
        // actual index of the function is.
        let entry =
            wasm_bindgen_wasm_conventions::get_function_table_entry(module, descriptor_table_idx)
                .expect("failed to find entry in function table");
        let descriptor_id = entry.func.expect("element segment slot wasn't set");
        entry_removal_list
            .entry(entry.element)
            .or_default()
            .insert(entry.idx);

        // And now execute the descriptor!
        self.interpret_descriptor(descriptor_id, module)
    }

    /// Returns the function id of the `__wbindgen_describe_closure`
    /// imported function.
    pub fn describe_closure_id(&self) -> Option<FunctionId> {
        self.describe_closure_id
    }

    /// Returns the detected id of the function table.
    pub fn function_table_id(&self) -> Option<TableId> {
        self.functions
    }

    fn call(&mut self, id: FunctionId, module: &Module, args: &[i32]) -> Option<i32> {
        let func = module.funcs.get(id);
        log::debug!("starting a call of {:?} {:?}", id, func.name);
        log::debug!("arguments {:?}", args);
        let local = match &func.kind {
            walrus::FunctionKind::Local(l) => l,
            _ => panic!("can only call locally defined functions"),
        };

        let entry = local.entry_block();
        let block = local.block(entry);

        let mut frame = Frame {
            module,
            interp: self,
            locals: BTreeMap::new(),
            done: false,
        };

        assert_eq!(local.args.len(), args.len());
        for (arg, val) in local.args.iter().zip(args) {
            frame.locals.insert(*arg, *val);
        }

        for (instr, _) in block.instrs.iter() {
            if let Some(name) = &module.funcs.get(id).name {
                log::trace!("eval {name} {instr:?}")
            }

            if let Err(err) = frame.eval(instr) {
                if let Some(name) = &module.funcs.get(id).name {
                    panic!("{name}: {err}")
                } else {
                    panic!("{err}")
                }
            }

            if frame.done {
                break;
            }
        }
        self.scratch.last().cloned()
    }
}

struct Frame<'a> {
    module: &'a Module,
    interp: &'a mut Interpreter,
    locals: BTreeMap<LocalId, i32>,
    done: bool,
}

impl Frame<'_> {
    fn eval(&mut self, instr: &Instr) -> anyhow::Result<()> {
        use walrus::ir::*;

        let stack = &mut self.interp.scratch;

        match instr {
            Instr::Const(c) => match c.value {
                Value::I32(n) => stack.push(n),
                _ => bail!("non-i32 constant"),
            },
            Instr::LocalGet(e) => stack.push(self.locals.get(&e.local).cloned().unwrap_or(0)),
            Instr::LocalSet(e) => {
                let val = stack.pop().unwrap();
                self.locals.insert(e.local, val);
            }
            Instr::LocalTee(e) => {
                let val = *stack.last().unwrap();
                self.locals.insert(e.local, val);
            }

            // Blindly assume all globals are the stack pointer
            Instr::GlobalGet(_) => stack.push(self.interp.sp),
            Instr::GlobalSet(_) => {
                let val = stack.pop().unwrap();
                self.interp.sp = val;
            }

            // Support simple arithmetic, mainly for the stack pointer
            // manipulation
            Instr::Binop(e) => {
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                stack.push(match e.op {
                    BinaryOp::I32Sub => lhs - rhs,
                    BinaryOp::I32Add => lhs + rhs,
                    op => bail!("invalid binary op {:?}", op),
                });
            }

            // Support small loads/stores to the stack. These show up in debug
            // mode where there's some traffic on the linear stack even when in
            // theory there doesn't need to be.
            Instr::Load(e) => {
                let address = stack.pop().unwrap();
                ensure!(
                    address >= 0,
                    "Read a negative address value from the stack. Did we run out of memory?"
                );
                let address = address as u32 + e.arg.offset;
                ensure!(address % 4 == 0);
                let address = address as usize / 4;
                stack.push(self.interp.mem[address])
            }
            Instr::Store(e) => {
                let value = stack.pop().unwrap();
                let address = stack.pop().unwrap();
                ensure!(
                    address > 0,
                    "Read a negative address value from the stack. Did we run out of memory?"
                );
                let address = address as u32 + e.arg.offset;
                ensure!(address % 4 == 0);
                self.interp.mem[address as usize / 4] = value;
            }

            Instr::Return(_) => {
                log::debug!("return");
                self.done = true;
            }

            Instr::Drop(_) => {
                log::debug!("drop");
                stack.pop().unwrap();
            }

            Instr::Call(Call { func }) | Instr::ReturnCall(ReturnCall { func }) => {
                let func = *func;
                // If this function is calling the `__wbindgen_describe`
                // function, which we've precomputed the id for, then
                // it's telling us about the next `u32` element in the
                // descriptor to return. We "call" the imported function
                // here by directly inlining it.
                if Some(func) == self.interp.describe_id {
                    let val = stack.pop().unwrap();
                    log::debug!("__wbindgen_describe({})", val);
                    self.interp.descriptor.push(val as u32);

                // If this function is calling the `__wbindgen_describe_closure`
                // function then it's similar to the above, except there's a
                // slightly different signature. Note that we don't eval the
                // previous arguments because they shouldn't have any side
                // effects we're interested in.
                } else if Some(func) == self.interp.describe_closure_id {
                    let val = stack.pop().unwrap();
                    stack.pop();
                    stack.pop();
                    log::debug!("__wbindgen_describe_closure({})", val);
                    self.interp.descriptor_table_idx = Some(val as u32);
                    stack.push(0)

                // ... otherwise this is a normal call so we recurse.
                } else {
                    // Skip profiling related functions which we don't want to interpret.
                    if self
                        .module
                        .funcs
                        .get(func)
                        .name
                        .as_ref()
                        .is_some_and(|name| {
                            name.starts_with("__llvm_profile_init")
                                || name.starts_with("__llvm_profile_register_function")
                                || name.starts_with("__llvm_profile_register_function")
                        })
                    {
                        return Ok(());
                    }

                    let ty = self.module.types.get(self.module.funcs.get(func).ty());
                    let args = (0..ty.params().len())
                        .map(|_| stack.pop().unwrap())
                        .collect::<Vec<_>>();

                    self.interp.call(func, self.module, &args);
                }

                if let Instr::ReturnCall(_) = instr {
                    log::debug!("return_call");
                    self.done = true;
                }
            }

            Instr::CallIndirect(CallIndirect { .. }) => {
                log::warn!("Ignoring {instr:?}");
            }
            //     let expected_ty_id = *ty;
            //     let table_id = *table;

            //     // 1. Pop the function index from the stack
            //     let func_index = stack.pop().unwrap();
            //     log::debug!(
            //         "call_indirect: type {:?}, table {:?}, index {}",
            //         expected_ty_id,
            //         table_id,
            //         func_index
            //     );

            //     // 2. Resolve the index in the table to a FunctionId
            //     //    NOTE: This requires a helper function `resolve_table_index`
            //     //          which needs access to `self.module` and handles element segments.
            //     //          A basic version is provided above.
            //     let target_func_id =
            //         Self::resolve_table_index(self.module, table_id, func_index as u32);

            //     if let Some(func) = target_func_id {
            //         // 3. Get the expected function type from the instruction
            //         let expected_ty = self.module.types.get(expected_ty_id);

            //         // --- Check for skipping profiling functions ---
            //         let func_data = self.module.funcs.get(func);
            //         if func_data.name.as_ref().is_some_and(|name| {
            //             name.starts_with("__llvm_profile_init")
            //                 || name.starts_with("__llvm_profile_register_function")
            //                 || name.starts_with("__llvm_profile_instrument_target")
            //         }) {
            //             log::debug!(
            //                 "Skipping indirect call to profiling function: {:?}",
            //                 func_data.name
            //             );
            //             // Pop args based on the EXPECTED type from call_indirect
            //             for _ in 0..expected_ty.params().len() {
            //                 stack.pop();
            //             }
            //             // Push dummy return values based on the EXPECTED type
            //             for _ in 0..expected_ty.results().len() {
            //                 stack.push(0); // Push default value (0)
            //             }

            //             // Handle ReturnCallIndirect similar to ReturnCall
            //             if let Instr::ReturnCallIndirect(_) = instr {
            //                 log::debug!("return_call_indirect after skipping profiling func");
            //                 self.done = true;
            //             }
            //             return Ok(()); // Skip the actual call
            //         }
            //         // --- End skipping check ---

            //         // 4. Pop arguments from the stack based on the EXPECTED type
            //         let mut args = (0..expected_ty.params().len())
            //             .map(|_| stack.pop().unwrap())
            //             .collect::<Vec<_>>();
            //         args.reverse(); // Popped in reverse order

            //         // 5. Recurse/call the resolved function
            //         log::debug!(
            //             "Calling func {:?} indirectly via index {} with args: {:?}",
            //             func,
            //             func_index,
            //             args
            //         );
            //         // TODO: Verify that the actual function's type matches expected_ty?
            //         // Wasm validation ensures this, but the interpreter might want to assert.
            //         // let actual_ty_id = self.module.funcs.get(func).ty();
            //         // assert_eq!(actual_ty_id, expected_ty_id, "Indirect call type mismatch!");

            //         self.interp.call(func, self.module, &args);
            //     } else {
            //         // Handle error: function index out of bounds or points to null
            //         // This should ideally trap in a real Wasm engine.
            //         // How should the interpreter handle this? Log an error? Panic?
            //         // Let's log and potentially push dummy return values if needed.
            //         log::error!(
            //             "CallIndirect failed: Could not resolve function index {} in table {:?}",
            //             func_index,
            //             table_id
            //         );
            //         // Pop arguments anyway based on the expected type
            //         let expected_ty = self.module.types.get(expected_ty_id);
            //         for _ in 0..expected_ty.params().len() {
            //             stack.pop();
            //         }
            //         // Push dummy return values if the call was expected to return something
            //         for _ in 0..expected_ty.results().len() {
            //             stack.push(0); // Push default value (0)
            //         }
            //         // If it was a ReturnCallIndirect, we still need to terminate this frame.
            //         // Or maybe it should panic? Depending on interpreter requirements.
            //         if let Instr::ReturnCallIndirect(_) = instr {
            //             log::debug!("return_call_indirect after failed resolution");
            //             self.done = true; // Or perhaps panic?
            //         }
            //         // Depending on strictness, you might want to return an error here:
            //         // return Err(anyhow::anyhow!("Indirect call resolution failed"));
            //     }

            //     // Handle ReturnCallIndirect similar to ReturnCall
            //     if let Instr::ReturnCallIndirect(_) = instr {
            //         log::debug!("return_call_indirect");
            //         self.done = true;
            //     }
            // }

            // All other instructions shouldn't be used by our various
            // descriptor functions. LLVM optimizations may mean that some
            // of the above instructions aren't actually needed either, but
            // the above instructions have empirically been required when
            // executing our own test suite in wasm-bindgen.
            //
            // Note that LLVM may change over time to generate new
            // instructions in debug mode, and we'll have to react to those
            // sorts of changes as they arise.
            s => bail!("unknown instruction {:?}", s),
        }

        Ok(())
    }

    fn resolve_table_index(module: &Module, table_id: TableId, index: u32) -> Option<FunctionId> {
        use log::{debug, warn};
        use walrus::{
            ir::Value, ConstExpr, ElementItems, ElementKind, FunctionId, GlobalId, GlobalKind,
            Module, RefType,
        };
        // Helper to resolve a ConstExpr (typically used for offsets) to a constant u32.
        // This might involve looking up global initializers.
        fn resolve_const_expr_to_u32(module: &Module, const_expr: &ConstExpr) -> Option<u32> {
            match const_expr {
                ConstExpr::Value(val) => match val {
                    Value::I32(c) => Some(*c as u32),
                    Value::I64(c) => Some(*c as u32), // Potential truncation, Wasm validates offset is i32
                    _ => {
                        warn!(
                            "Unsupported ConstExpr::Value type for table offset: {:?}",
                            val
                        );
                        None
                    }
                },
                ConstExpr::Global(global_id) => resolve_global_init_to_u32(module, *global_id),
                // Other ConstExpr variants are not valid for constant table offsets according to Wasm validation rules.
                _ => {
                    warn!(
                        "Unsupported ConstExpr type for table offset: {:?}",
                        const_expr
                    );
                    None
                }
            }
        }

        // Helper function to resolve a global's initial value to u32.
        // Note: Globals are initialized with InitExpr, not ConstExpr.
        fn resolve_global_init_to_u32(module: &Module, global_id: GlobalId) -> Option<u32> {
            let global = module.globals.get(global_id);
            match global.kind {
                GlobalKind::Local(ref const_expr) => {
                    // Resolve the ConstExpr used for the global's initialization.
                    resolve_init_expr_to_u32(module, const_expr)
                }
                GlobalKind::Import(_) => {
                    warn!(
                        "Cannot statically resolve table offset: Global {:?} is imported.",
                        global_id
                    );
                    None
                }
            }
        }

        // Helper function to resolve an InitExpr to a constant u32.
        // Needed because Globals use InitExpr for their definition.
        fn resolve_init_expr_to_u32(module: &Module, init_expr: &ConstExpr) -> Option<u32> {
            match init_expr {
                ConstExpr::Value(val) => match val {
                    Value::I32(c) => Some(*c as u32),
                    Value::I64(c) => Some(*c as u32), // Truncation
                    _ => {
                        warn!(
                            "Unsupported InitExpr::Value type for global used as offset: {:?}",
                            val
                        );
                        None
                    }
                },
                ConstExpr::Global(global_id) => {
                    // Recursively resolve the global, checking for cycles is implicitly needed by caller or Wasm validation.
                    resolve_global_init_to_u32(module, *global_id)
                }
                _ => {
                    warn!(
                        "Unsupported InitExpr type for global used as offset: {:?}",
                        init_expr
                    );
                    None
                }
            }
        }

        // Iterate through all element segments in the module
        for element in module.elements.iter() {
            // We only care about Active segments that initialize tables.
            if let ElementKind::Active {
                table: element_table_id,       // The TableId being initialized
                offset: ref const_expr_offset, // The ConstExpr defining the offset
            } = element.kind
            {
                // Check if this segment targets the table we are interested in
                if element_table_id == table_id {
                    // Try to resolve the offset expression to a constant value
                    if let Some(base_offset) = resolve_const_expr_to_u32(module, const_expr_offset)
                    {
                        let (items_len, get_func_id_at_item_index): (
                            usize,
                            Box<dyn Fn(usize) -> Option<FunctionId>>,
                        ) = match &element.items {
                            ElementItems::Functions(functions) => {
                                // Directly contains FunctionIds
                                (
                                    functions.len(),
                                    Box::new(move |item_idx| functions.get(item_idx).copied()),
                                )
                            }
                            ElementItems::Expressions(ref_ty, expressions) => {
                                // Contains expressions, need to check for RefFunc
                                if ref_ty != &RefType::Funcref {
                                    // This segment initializes the table with something other than functions.
                                    // Skip this segment for function resolution.
                                    debug!(
                                            "Skipping element {:?} for table {:?}: items are expressions of type {:?}, not funcref.",
                                            element.id(), table_id, ref_ty
                                        );
                                    continue; // Move to the next element segment
                                }
                                (
                                    expressions.len(),
                                    Box::new(move |item_idx| {
                                        match expressions.get(item_idx) {
                                            Some(ConstExpr::RefFunc(func_id)) => Some(*func_id),
                                            Some(ConstExpr::RefNull(_)) => None, // Explicit null reference
                                            Some(other_expr) => {
                                                // This shouldn't happen if ref_ty is FuncRef, but handle defensively.
                                                warn!(
                                                        "Unexpected expression {:?} found in funcref element segment {:?} at item index {}",
                                                        other_expr, element.id(), item_idx
                                                    );
                                                None
                                            }
                                            None => None, // Index out of bounds for expressions vec
                                        }
                                    }),
                                )
                            }
                        };

                        let items_len_u32 = items_len as u32;

                        // Check if the requested index falls within the range covered by this segment
                        if index >= base_offset && index < base_offset + items_len_u32 {
                            // Calculate the index within the segment's items
                            let item_index = (index - base_offset) as usize;

                            // Use the appropriate closure to get the FunctionId (or None)
                            let func_id_option = get_func_id_at_item_index(item_index);

                            debug!(
                                "Resolved index {} in table {:?} via element {:?} (offset {}): {:?}",
                                index,
                                table_id,
                                element.id(),
                                base_offset,
                                func_id_option
                            );
                            // Return the result (which could be Some(FunctionId) or None)
                            return func_id_option;
                        }
                    } else {
                        // Offset couldn't be resolved statically (e.g., imported global)
                        warn!(
                            "Skipping element segment {:?} for table {:?} due to unresolvable offset expression: {:?}",
                            element.id(),
                            table_id,
                            const_expr_offset
                        );
                    }
                }
            }
            // Ignore Passive and Declared segments as they don't auto-initialize tables.
        }

        // If no active segment covering the index was found or resolved
        debug!(
            "Index {} in table {:?} not found in any resolvable active element segment.",
            index, table_id
        );
        None
    }
}

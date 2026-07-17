"""Canonical generated Rust module ``rextio_tensorflow_runtime``.

The Alpha runtime deliberately binds only the already-loaded TensorFlow 2.21.0
wheel.  It owns eager tensor-handle references with RAII while keeping the
Python eager context borrowed and strongly anchored by Python references.
"""

from __future__ import annotations

# Single exact-text helper block. Core deduplicates it across PluginType and
# LoweredExpr support collectors.
_RUNTIME_MODULE = r"""
mod rextio_tensorflow_runtime {
    use pyo3::prelude::*;
    use pyo3::types::PyAny;
    use std::ffi::{CStr, CString};
    use std::marker::PhantomData;
    use std::os::raw::{c_char, c_int, c_void};
    use std::path::{Path, PathBuf};
    use std::rc::Rc;
    use std::sync::OnceLock;

    type TfStatus = c_void;
    type TfTensor = c_void;
    type TfeContext = c_void;
    type TfeTensorHandle = c_void;
    type TfeOp = c_void;

    const TF_FLOAT: c_int = 1;
    const TF_INT32: c_int = 3;
    const EXPECTED_TF_VERSION: &str = "2.21.0";

    // Darwin dlfcn flags. RTLD_NOLOAD is the essential same-runtime gate: a
    // missing image is an error, never an instruction to load another copy.
    const RTLD_NOW: c_int = 0x2;
    const RTLD_LOCAL: c_int = 0x4;
    const RTLD_NOLOAD: c_int = 0x10;
    const TF_DLOPEN_FLAGS: c_int = RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD;

    // Private ABI bridge (Itanium mangling in the TF 2.21.0 macOS arm64 wheel).
    const SYM_EAGER_TENSOR_HANDLE: &str = "_Z18EagerTensor_HandlePK7_object";
    const SYM_EAGER_TENSOR_FROM_HANDLE: &str =
        "_Z21EagerTensorFromHandleP16TFE_TensorHandleb";
    const SYM_EAGER_TENSOR_CHECK_EXACT: &str =
        "_Z22EagerTensor_CheckExactPK7_object";

    #[repr(C)]
    struct DlInfo {
        dli_fname: *const c_char,
        dli_fbase: *mut c_void,
        dli_sname: *const c_char,
        dli_saddr: *mut c_void,
    }

    unsafe extern "C" {
        fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
        fn dlerror() -> *const c_char;
        fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
    }

    fn runtime_error(message: impl Into<String>) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "rextio-tensorflow: {}",
            message.into()
        ))
    }

    fn value_error(message: impl Into<String>) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(format!(
            "rextio-tensorflow: {}",
            message.into()
        ))
    }

    fn c_string(value: &str, label: &str) -> PyResult<CString> {
        CString::new(value).map_err(|_| runtime_error(format!("{label} contains NUL")))
    }

    fn canonicalize(path: &Path, label: &str) -> PyResult<PathBuf> {
        path.canonicalize().map_err(|error| {
            runtime_error(format!(
                "cannot canonicalize {label} {}: {error}",
                path.display()
            ))
        })
    }

    fn active_tensorflow_root(py: Python<'_>) -> PyResult<PathBuf> {
        let tensorflow = py.import("tensorflow")?;
        let python_version: String = tensorflow.getattr("__version__")?.extract()?;
        if python_version != EXPECTED_TF_VERSION {
            return Err(runtime_error(format!(
                "Python TensorFlow version mismatch: expected {EXPECTED_TF_VERSION}, got {python_version}"
            )));
        }
        let module_file: String = tensorflow.getattr("__file__")?.extract()?;
        let canonical_file = canonicalize(Path::new(&module_file), "tensorflow.__file__")?;
        canonical_file
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| runtime_error("tensorflow.__file__ has no package parent"))
    }

    fn validate_platform(py: Python<'_>) -> PyResult<()> {
        if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return Err(runtime_error(
                "Alpha requires aarch64-apple-darwin generated Rust",
            ));
        }
        let sys = py.import("sys")?;
        let implementation: String = sys
            .getattr("implementation")?
            .getattr("name")?
            .extract()?;
        let version = sys.getattr("version_info")?;
        let major: i64 = version.getattr("major")?.extract()?;
        let minor: i64 = version.getattr("minor")?.extract()?;
        if implementation != "cpython" || (major, minor) != (3, 11) {
            return Err(runtime_error(format!(
                "Alpha requires CPython 3.11, got {implementation} {major}.{minor}"
            )));
        }
        let machine: String = py.import("platform")?.call_method0("machine")?.extract()?;
        if machine != "arm64" {
            return Err(runtime_error(format!(
                "Alpha requires macOS arm64, got machine={machine}"
            )));
        }
        Ok(())
    }

    fn dl_error_text() -> String {
        let pointer = unsafe { dlerror() };
        if pointer.is_null() {
            "unknown dynamic-loader error".to_string()
        } else {
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        }
    }

    struct DlHandle {
        raw: *mut c_void,
        canonical_path: PathBuf,
    }

    impl DlHandle {
        fn open_noload(path: PathBuf, label: &str) -> PyResult<Self> {
            let path_text = path
                .to_str()
                .ok_or_else(|| runtime_error(format!("{label} path is not UTF-8")))?;
            let c_path = c_string(path_text, label)?;
            unsafe {
                let _ = dlerror();
            }
            let raw = unsafe { dlopen(c_path.as_ptr(), TF_DLOPEN_FLAGS) };
            if raw.is_null() {
                return Err(runtime_error(format!(
                    "active {label} image is not already loaded at {}: {}",
                    path.display(),
                    dl_error_text()
                )));
            }
            Ok(Self {
                raw,
                canonical_path: path,
            })
        }

        unsafe fn resolve<T>(&self, symbol: &str) -> PyResult<T>
        where
            T: Copy,
        {
            if std::mem::size_of::<T>() != std::mem::size_of::<*mut c_void>() {
                return Err(runtime_error(format!(
                    "unexpected function-pointer size for {symbol}"
                )));
            }
            let c_symbol = c_string(symbol, "symbol name")?;
            let _ = dlerror();
            let pointer = dlsym(self.raw, c_symbol.as_ptr());
            if pointer.is_null() {
                return Err(runtime_error(format!(
                    "required TensorFlow C/Eager symbol missing: {symbol}: {}",
                    dl_error_text()
                )));
            }
            verify_provenance(pointer.cast_const(), &self.canonical_path, symbol)?;
            Ok(std::mem::transmute_copy::<*mut c_void, T>(&pointer))
        }

        fn retain(mut self) -> usize {
            let raw = self.raw as usize;
            self.raw = std::ptr::null_mut();
            raw
        }
    }

    impl Drop for DlHandle {
        fn drop(&mut self) {
            if !self.raw.is_null() {
                unsafe {
                    let _ = dlclose(self.raw);
                }
                self.raw = std::ptr::null_mut();
            }
        }
    }

    fn verify_provenance(
        address: *const c_void,
        expected_path: &Path,
        symbol: &str,
    ) -> PyResult<()> {
        let mut info = DlInfo {
            dli_fname: std::ptr::null(),
            dli_fbase: std::ptr::null_mut(),
            dli_sname: std::ptr::null(),
            dli_saddr: std::ptr::null_mut(),
        };
        let found = unsafe { dladdr(address, &mut info) };
        if found == 0 || info.dli_fname.is_null() {
            return Err(runtime_error(format!(
                "dladdr could not prove provenance for {symbol}"
            )));
        }
        let reported = unsafe { CStr::from_ptr(info.dli_fname) }
            .to_string_lossy()
            .into_owned();
        let actual_path = canonicalize(Path::new(&reported), "dladdr image")?;
        if actual_path != expected_path {
            return Err(runtime_error(format!(
                "symbol provenance mismatch for {symbol}: expected {}, got {}",
                expected_path.display(),
                actual_path.display()
            )));
        }
        Ok(())
    }

    #[allow(dead_code)]
    struct Api {
        // Retained RTLD_NOLOAD handles keep all resolved function pointers live.
        cc_handle: usize,
        framework_handle: usize,
        pywrap_handle: usize,
        tensorflow_root: PathBuf,
        cc_path: PathBuf,
        framework_path: PathBuf,
        pywrap_path: PathBuf,

        tf_version: unsafe extern "C" fn() -> *const c_char,
        tf_new_status: unsafe extern "C" fn() -> *mut TfStatus,
        tf_delete_status: unsafe extern "C" fn(*mut TfStatus),
        tf_get_code: unsafe extern "C" fn(*const TfStatus) -> c_int,
        tf_message: unsafe extern "C" fn(*const TfStatus) -> *const c_char,
        tf_allocate_tensor:
            unsafe extern "C" fn(c_int, *const i64, c_int, usize) -> *mut TfTensor,
        tf_delete_tensor: unsafe extern "C" fn(*mut TfTensor),
        tf_tensor_data: unsafe extern "C" fn(*const TfTensor) -> *mut c_void,
        tf_tensor_byte_size: unsafe extern "C" fn(*const TfTensor) -> usize,

        tfe_new_tensor_handle:
            unsafe extern "C" fn(*const TfTensor, *mut TfStatus) -> *mut TfeTensorHandle,
        tfe_delete_tensor_handle: unsafe extern "C" fn(*mut TfeTensorHandle),
        tfe_tensor_handle_copy_sharing:
            unsafe extern "C" fn(*mut TfeTensorHandle, *mut TfStatus) -> *mut TfeTensorHandle,
        tfe_tensor_handle_data_type: unsafe extern "C" fn(*mut TfeTensorHandle) -> c_int,
        tfe_tensor_handle_num_dims:
            unsafe extern "C" fn(*mut TfeTensorHandle, *mut TfStatus) -> c_int,
        tfe_tensor_handle_dim:
            unsafe extern "C" fn(*mut TfeTensorHandle, c_int, *mut TfStatus) -> i64,
        tfe_tensor_handle_backing_device_name:
            unsafe extern "C" fn(*mut TfeTensorHandle, *mut TfStatus) -> *const c_char,

        tfe_new_op:
            unsafe extern "C" fn(*mut TfeContext, *const c_char, *mut TfStatus) -> *mut TfeOp,
        tfe_delete_op: unsafe extern "C" fn(*mut TfeOp),
        tfe_op_set_device:
            unsafe extern "C" fn(*mut TfeOp, *const c_char, *mut TfStatus),
        tfe_op_add_input:
            unsafe extern "C" fn(*mut TfeOp, *mut TfeTensorHandle, *mut TfStatus),
        tfe_op_set_attr_type: unsafe extern "C" fn(*mut TfeOp, *const c_char, c_int),
        tfe_op_set_attr_bool: unsafe extern "C" fn(*mut TfeOp, *const c_char, u8),
        tfe_execute: unsafe extern "C" fn(
            *mut TfeOp,
            *mut *mut TfeTensorHandle,
            *mut c_int,
            *mut TfStatus,
        ),

        eager_tensor_handle:
            unsafe extern "C" fn(*const pyo3::ffi::PyObject) -> *mut TfeTensorHandle,
        eager_tensor_from_handle:
            unsafe extern "C" fn(*mut TfeTensorHandle, bool) -> *mut pyo3::ffi::PyObject,
        eager_tensor_check_exact:
            unsafe extern "C" fn(*const pyo3::ffi::PyObject) -> bool,
    }

    impl Api {
        fn load(py: Python<'_>) -> PyResult<Self> {
            validate_platform(py)?;
            let tensorflow_root = active_tensorflow_root(py)?;
            let cc_path = canonicalize(
                &tensorflow_root.join("libtensorflow_cc.2.dylib"),
                "libtensorflow_cc",
            )?;
            let framework_path = canonicalize(
                &tensorflow_root.join("libtensorflow_framework.2.dylib"),
                "libtensorflow_framework",
            )?;
            let pywrap_path = canonicalize(
                &tensorflow_root.join("python/lib_pywrap_tensorflow_common.dylib"),
                "lib_pywrap_tensorflow_common",
            )?;
            for (path, label) in [
                (&cc_path, "libtensorflow_cc"),
                (&framework_path, "libtensorflow_framework"),
                (&pywrap_path, "lib_pywrap_tensorflow_common"),
            ] {
                if !path.starts_with(&tensorflow_root) {
                    return Err(runtime_error(format!(
                        "{label} resolved outside the active TensorFlow wheel: {}",
                        path.display()
                    )));
                }
            }

            let cc = DlHandle::open_noload(cc_path.clone(), "libtensorflow_cc")?;
            let framework =
                DlHandle::open_noload(framework_path.clone(), "libtensorflow_framework")?;
            let pywrap = DlHandle::open_noload(
                pywrap_path.clone(),
                "lib_pywrap_tensorflow_common",
            )?;

            let api = unsafe {
                Self {
                    cc_handle: 0,
                    framework_handle: 0,
                    pywrap_handle: 0,
                    tensorflow_root,
                    cc_path,
                    framework_path,
                    pywrap_path,

                    tf_version: cc.resolve("TF_Version")?,
                    tf_new_status: framework.resolve("TF_NewStatus")?,
                    tf_delete_status: framework.resolve("TF_DeleteStatus")?,
                    tf_get_code: framework.resolve("TF_GetCode")?,
                    tf_message: framework.resolve("TF_Message")?,
                    tf_allocate_tensor: framework.resolve("TF_AllocateTensor")?,
                    tf_delete_tensor: framework.resolve("TF_DeleteTensor")?,
                    tf_tensor_data: framework.resolve("TF_TensorData")?,
                    tf_tensor_byte_size: framework.resolve("TF_TensorByteSize")?,

                    tfe_new_tensor_handle: cc.resolve("TFE_NewTensorHandle")?,
                    tfe_delete_tensor_handle: cc.resolve("TFE_DeleteTensorHandle")?,
                    tfe_tensor_handle_copy_sharing: cc.resolve(
                        "TFE_TensorHandleCopySharingTensor",
                    )?,
                    tfe_tensor_handle_data_type: cc.resolve("TFE_TensorHandleDataType")?,
                    tfe_tensor_handle_num_dims: cc.resolve("TFE_TensorHandleNumDims")?,
                    tfe_tensor_handle_dim: cc.resolve("TFE_TensorHandleDim")?,
                    tfe_tensor_handle_backing_device_name: cc.resolve(
                        "TFE_TensorHandleBackingDeviceName",
                    )?,
                    tfe_new_op: cc.resolve("TFE_NewOp")?,
                    tfe_delete_op: cc.resolve("TFE_DeleteOp")?,
                    tfe_op_set_device: cc.resolve("TFE_OpSetDevice")?,
                    tfe_op_add_input: cc.resolve("TFE_OpAddInput")?,
                    tfe_op_set_attr_type: cc.resolve("TFE_OpSetAttrType")?,
                    tfe_op_set_attr_bool: cc.resolve("TFE_OpSetAttrBool")?,
                    tfe_execute: cc.resolve("TFE_Execute")?,

                    eager_tensor_handle: pywrap.resolve(SYM_EAGER_TENSOR_HANDLE)?,
                    eager_tensor_from_handle: pywrap.resolve(SYM_EAGER_TENSOR_FROM_HANDLE)?,
                    eager_tensor_check_exact: pywrap.resolve(SYM_EAGER_TENSOR_CHECK_EXACT)?,
                }
            };

            let version_pointer = unsafe { (api.tf_version)() };
            if version_pointer.is_null() {
                return Err(runtime_error("TF_Version returned null"));
            }
            let c_version = unsafe { CStr::from_ptr(version_pointer) }
                .to_string_lossy()
                .into_owned();
            if c_version != EXPECTED_TF_VERSION {
                return Err(runtime_error(format!(
                    "TensorFlow C runtime version mismatch: expected {EXPECTED_TF_VERSION}, got {c_version}"
                )));
            }

            let mut retained = api;
            retained.cc_handle = cc.retain();
            retained.framework_handle = framework.retain();
            retained.pywrap_handle = pywrap.retain();
            Ok(retained)
        }

        fn validate_active_wheel(&self, py: Python<'_>) -> PyResult<()> {
            let root = active_tensorflow_root(py)?;
            if root != self.tensorflow_root {
                return Err(runtime_error(format!(
                    "active TensorFlow wheel changed: expected {}, got {}",
                    self.tensorflow_root.display(),
                    root.display()
                )));
            }
            Ok(())
        }
    }

    impl Drop for Api {
        fn drop(&mut self) {
            for raw in [self.pywrap_handle, self.cc_handle, self.framework_handle] {
                if raw != 0 {
                    unsafe {
                        let _ = dlclose(raw as *mut c_void);
                    }
                }
            }
            self.pywrap_handle = 0;
            self.cc_handle = 0;
            self.framework_handle = 0;
        }
    }

    static API: OnceLock<Api> = OnceLock::new();

    fn load_api(py: Python<'_>) -> PyResult<&'static Api> {
        if let Some(api) = API.get() {
            api.validate_active_wheel(py)?;
            return Ok(api);
        }
        let candidate = Api::load(py)?;
        if let Err(unused) = API.set(candidate) {
            drop(unused);
        }
        let api = API
            .get()
            .ok_or_else(|| runtime_error("failed to retain TensorFlow API table"))?;
        api.validate_active_wheel(py)?;
        Ok(api)
    }

    struct OwnedStatus {
        api: &'static Api,
        raw: *mut TfStatus,
    }

    impl OwnedStatus {
        fn new(api: &'static Api) -> PyResult<Self> {
            let raw = unsafe { (api.tf_new_status)() };
            if raw.is_null() {
                return Err(runtime_error("TF_NewStatus returned null"));
            }
            Ok(Self { api, raw })
        }

        fn pointer(&self) -> *mut TfStatus {
            self.raw
        }

        fn check(&self, step: &str) -> PyResult<()> {
            let code = unsafe { (self.api.tf_get_code)(self.raw.cast_const()) };
            if code == 0 {
                return Ok(());
            }
            let message_pointer = unsafe { (self.api.tf_message)(self.raw.cast_const()) };
            let message = if message_pointer.is_null() {
                "<null TF_Message>".to_string()
            } else {
                unsafe { CStr::from_ptr(message_pointer) }
                    .to_string_lossy()
                    .into_owned()
            };
            Err(runtime_error(format!(
                "{step}: TensorFlow status code={code}: {message}"
            )))
        }
    }

    impl Drop for OwnedStatus {
        fn drop(&mut self) {
            if !self.raw.is_null() {
                unsafe { (self.api.tf_delete_status)(self.raw) };
                self.raw = std::ptr::null_mut();
            }
        }
    }

    struct BorrowedContext {
        api: &'static Api,
        raw: *mut TfeContext,
        _python_context: Py<PyAny>,
        _python_capsule: Py<PyAny>,
        _thread_affine: PhantomData<Rc<()>>,
    }

    impl BorrowedContext {
        fn from_python(py: Python<'_>, api: &'static Api) -> PyResult<Rc<Self>> {
            api.validate_active_wheel(py)?;
            let eager = py.import("tensorflow.python.eager.context")?;
            let context = eager.call_method0("context")?;
            context.call_method0("ensure_initialized")?;
            let is_async: bool = context.call_method0("is_async")?.extract()?;
            if is_async {
                return Err(runtime_error(
                    "Alpha requires the existing synchronous Python eager context",
                ));
            }
            let capsule = context.getattr("_handle")?;
            let capsule_name = unsafe { pyo3::ffi::PyCapsule_GetName(capsule.as_ptr()) };
            if unsafe { !pyo3::ffi::PyErr_Occurred().is_null() } {
                return Err(PyErr::fetch(py));
            }
            if !capsule_name.is_null() {
                let name = unsafe { CStr::from_ptr(capsule_name) }.to_string_lossy();
                return Err(runtime_error(format!(
                    "unexpected named eager context capsule {name:?}"
                )));
            }
            let raw = unsafe {
                pyo3::ffi::PyCapsule_GetPointer(capsule.as_ptr(), std::ptr::null())
            };
            if raw.is_null() {
                if unsafe { !pyo3::ffi::PyErr_Occurred().is_null() } {
                    return Err(PyErr::fetch(py));
                }
                return Err(runtime_error("Python eager context capsule is null"));
            }
            Ok(Rc::new(Self {
                api,
                raw: raw.cast::<TfeContext>(),
                _python_context: context.unbind(),
                _python_capsule: capsule.unbind(),
                _thread_affine: PhantomData,
            }))
        }
    }

    struct PendingHandle {
        api: &'static Api,
        raw: *mut TfeTensorHandle,
    }

    impl PendingHandle {
        fn new(api: &'static Api, raw: *mut TfeTensorHandle) -> Self {
            Self { api, raw }
        }

        fn into_raw(mut self) -> *mut TfeTensorHandle {
            let raw = self.raw;
            self.raw = std::ptr::null_mut();
            raw
        }

        fn into_owned(
            self,
            context: Rc<BorrowedContext>,
        ) -> PyResult<Rc<OwnedTensorHandle>> {
            if self.raw.is_null() {
                return Err(runtime_error("TensorFlow returned a null tensor handle"));
            }
            let api = self.api;
            let raw = self.into_raw();
            Ok(Rc::new(OwnedTensorHandle {
                api,
                raw,
                context,
                _thread_affine: PhantomData,
            }))
        }
    }

    impl Drop for PendingHandle {
        fn drop(&mut self) {
            if !self.raw.is_null() {
                unsafe { (self.api.tfe_delete_tensor_handle)(self.raw) };
                self.raw = std::ptr::null_mut();
            }
        }
    }

    struct OwnedTensorHandle {
        api: &'static Api,
        raw: *mut TfeTensorHandle,
        // The Python-owned context must outlive every eager tensor handle.
        context: Rc<BorrowedContext>,
        _thread_affine: PhantomData<Rc<()>>,
    }

    impl Drop for OwnedTensorHandle {
        fn drop(&mut self) {
            if !self.raw.is_null() {
                unsafe { (self.api.tfe_delete_tensor_handle)(self.raw) };
                self.raw = std::ptr::null_mut();
            }
        }
    }

    /// Clone shares one Rust owner. It never performs a fallible C call and
    /// never creates a non-owning fallback alias.
    #[derive(Clone)]
    pub struct RxtTfTensor {
        inner: Rc<OwnedTensorHandle>,
    }

    impl RxtTfTensor {
        fn from_pending(
            pending: PendingHandle,
            context: Rc<BorrowedContext>,
        ) -> PyResult<Self> {
            Ok(Self {
                inner: pending.into_owned(context)?,
            })
        }

        fn pointer(&self) -> *mut TfeTensorHandle {
            self.inner.raw
        }

        fn context(&self) -> Rc<BorrowedContext> {
            Rc::clone(&self.inner.context)
        }

        fn copy_sharing(&self) -> PyResult<PendingHandle> {
            let status = OwnedStatus::new(self.inner.api)?;
            let raw = unsafe {
                (self.inner.api.tfe_tensor_handle_copy_sharing)(
                    self.inner.raw,
                    status.pointer(),
                )
            };
            let pending = PendingHandle::new(self.inner.api, raw);
            status.check("TFE_TensorHandleCopySharingTensor")?;
            if pending.raw.is_null() {
                return Err(runtime_error(
                    "TFE_TensorHandleCopySharingTensor returned null",
                ));
            }
            Ok(pending)
        }

        fn backing_device(&self) -> PyResult<String> {
            let status = OwnedStatus::new(self.inner.api)?;
            let pointer = unsafe {
                (self.inner.api.tfe_tensor_handle_backing_device_name)(
                    self.inner.raw,
                    status.pointer(),
                )
            };
            status.check("TFE_TensorHandleBackingDeviceName")?;
            if pointer.is_null() {
                return Err(value_error("expected a CPU tensor"));
            }
            let device = unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned();
            if !device.ends_with("/device:CPU:0") {
                return Err(value_error(format!(
                    "expected a CPU:0 tensor, got device {device}"
                )));
            }
            Ok(device)
        }

        fn rank(&self) -> PyResult<c_int> {
            let status = OwnedStatus::new(self.inner.api)?;
            let rank = unsafe {
                (self.inner.api.tfe_tensor_handle_num_dims)(
                    self.inner.raw,
                    status.pointer(),
                )
            };
            status.check("TFE_TensorHandleNumDims")?;
            Ok(rank)
        }

        fn validate(&self, expected_rank: c_int) -> PyResult<()> {
            if unsafe { (self.inner.api.tfe_tensor_handle_data_type)(self.inner.raw) }
                != TF_FLOAT
            {
                return Err(value_error("expected a float32 tensor"));
            }
            let rank = self.rank()?;
            if rank != expected_rank {
                return Err(value_error(format!(
                    "expected rank-{expected_rank} tensor, got rank {rank}"
                )));
            }
            let status = OwnedStatus::new(self.inner.api)?;
            for index in 0..rank {
                let dimension = unsafe {
                    (self.inner.api.tfe_tensor_handle_dim)(
                        self.inner.raw,
                        index,
                        status.pointer(),
                    )
                };
                status.check("TFE_TensorHandleDim")?;
                if dimension < 0 {
                    return Err(value_error(format!(
                        "expected concrete eager dimensions, got dim[{index}]={dimension}"
                    )));
                }
            }
            let _ = self.backing_device()?;
            Ok(())
        }
    }

    struct OwnedTfTensor {
        api: &'static Api,
        raw: *mut TfTensor,
    }

    impl OwnedTfTensor {
        fn axis_one(api: &'static Api) -> PyResult<Self> {
            let dimensions = [1i64];
            let raw = unsafe {
                (api.tf_allocate_tensor)(
                    TF_INT32,
                    dimensions.as_ptr(),
                    1,
                    std::mem::size_of::<i32>(),
                )
            };
            if raw.is_null() {
                return Err(runtime_error("TF_AllocateTensor failed for mean axis"));
            }
            let tensor = Self { api, raw };
            let byte_size = unsafe { (api.tf_tensor_byte_size)(tensor.raw.cast_const()) };
            if byte_size != std::mem::size_of::<i32>() {
                return Err(runtime_error(format!(
                    "unexpected mean-axis tensor byte size {byte_size}"
                )));
            }
            let data = unsafe { (api.tf_tensor_data)(tensor.raw.cast_const()) };
            if data.is_null() {
                return Err(runtime_error("TF_TensorData returned null for mean axis"));
            }
            unsafe {
                std::ptr::write(data.cast::<i32>(), 1i32);
            }
            Ok(tensor)
        }
    }

    impl Drop for OwnedTfTensor {
        fn drop(&mut self) {
            if !self.raw.is_null() {
                unsafe { (self.api.tf_delete_tensor)(self.raw) };
                self.raw = std::ptr::null_mut();
            }
        }
    }

    struct OwnedOp {
        api: &'static Api,
        raw: *mut TfeOp,
        _context: Rc<BorrowedContext>,
    }

    impl OwnedOp {
        fn new(
            context: Rc<BorrowedContext>,
            op_name: &str,
            status: &OwnedStatus,
        ) -> PyResult<Self> {
            let name = c_string(op_name, "TensorFlow op name")?;
            let raw = unsafe {
                (context.api.tfe_new_op)(context.raw, name.as_ptr(), status.pointer())
            };
            let op = Self {
                api: context.api,
                raw,
                _context: context,
            };
            status.check(&format!("TFE_NewOp({op_name})"))?;
            if op.raw.is_null() {
                return Err(runtime_error(format!(
                    "TFE_NewOp({op_name}) returned null"
                )));
            }
            Ok(op)
        }

        fn set_device(&self, device: &str, status: &OwnedStatus) -> PyResult<()> {
            let c_device = c_string(device, "TensorFlow device")?;
            unsafe {
                (self.api.tfe_op_set_device)(self.raw, c_device.as_ptr(), status.pointer());
            }
            status.check("TFE_OpSetDevice")
        }

        fn add_input(&self, input: *mut TfeTensorHandle, status: &OwnedStatus) -> PyResult<()> {
            unsafe {
                (self.api.tfe_op_add_input)(self.raw, input, status.pointer());
            }
            status.check("TFE_OpAddInput")
        }

        fn set_type(&self, name: &str, value: c_int) -> PyResult<()> {
            let c_name = c_string(name, "TensorFlow type attribute")?;
            unsafe { (self.api.tfe_op_set_attr_type)(self.raw, c_name.as_ptr(), value) };
            Ok(())
        }

        fn set_bool(&self, name: &str, value: bool) -> PyResult<()> {
            let c_name = c_string(name, "TensorFlow bool attribute")?;
            unsafe {
                (self.api.tfe_op_set_attr_bool)(
                    self.raw,
                    c_name.as_ptr(),
                    if value { 1 } else { 0 },
                )
            };
            Ok(())
        }

        fn execute_one(&self, status: &OwnedStatus) -> PyResult<PendingHandle> {
            let mut output = std::ptr::null_mut();
            let mut output_count: c_int = 1;
            unsafe {
                (self.api.tfe_execute)(
                    self.raw,
                    &mut output,
                    &mut output_count,
                    status.pointer(),
                );
            }
            // Construct cleanup ownership before inspecting either status or count.
            let pending = PendingHandle::new(self.api, output);
            status.check("TFE_Execute")?;
            if output_count != 1 {
                return Err(runtime_error(format!(
                    "TFE_Execute returned {output_count} outputs; expected exactly 1"
                )));
            }
            if pending.raw.is_null() {
                return Err(runtime_error(
                    "TFE_Execute returned a null single output",
                ));
            }
            Ok(pending)
        }
    }

    impl Drop for OwnedOp {
        fn drop(&mut self) {
            if !self.raw.is_null() {
                unsafe { (self.api.tfe_delete_op)(self.raw) };
                self.raw = std::ptr::null_mut();
            }
        }
    }

    fn same_context(left: &RxtTfTensor, right: &RxtTfTensor) -> PyResult<()> {
        if left.inner.context.raw != right.inner.context.raw
            || !std::ptr::eq(left.inner.api, right.inner.api)
        {
            return Err(runtime_error(
                "tensor operands do not belong to the same Python eager context",
            ));
        }
        Ok(())
    }

    fn python_error_or(py: Python<'_>, fallback: &str) -> PyErr {
        if unsafe { !pyo3::ffi::PyErr_Occurred().is_null() } {
            PyErr::fetch(py)
        } else {
            runtime_error(fallback)
        }
    }

    fn extract_common(
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
        expected_rank: c_int,
    ) -> PyResult<RxtTfTensor> {
        let api = load_api(py)?;
        let context = BorrowedContext::from_python(py, api)?;
        let object = value.as_ptr().cast_const();
        // Both private bridge calls are made only while this Python token holds
        // the GIL. Never move them into Python::detach / allow_threads code.
        if unsafe { !(api.eager_tensor_check_exact)(object) } {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "rextio-tensorflow: expected a TensorFlow EagerTensor",
            ));
        }
        let borrowed = unsafe { (api.eager_tensor_handle)(object) };
        if borrowed.is_null() {
            return Err(python_error_or(
                py,
                "EagerTensor_Handle returned null",
            ));
        }
        let status = OwnedStatus::new(api)?;
        let raw = unsafe {
            (api.tfe_tensor_handle_copy_sharing)(borrowed, status.pointer())
        };
        let pending = PendingHandle::new(api, raw);
        status.check("TFE_TensorHandleCopySharingTensor(input)")?;
        let tensor = RxtTfTensor::from_pending(pending, context)?;
        tensor.validate(expected_rank)?;
        Ok(tensor)
    }

    pub fn extract_f32_cpu_2d(
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<RxtTfTensor> {
        extract_common(py, value, 2)
    }

    pub fn extract_f32_cpu_1d(
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<RxtTfTensor> {
        extract_common(py, value, 1)
    }

    pub fn materialize_tensor(
        py: Python<'_>,
        value: RxtTfTensor,
    ) -> PyResult<Bound<'_, PyAny>> {
        value.inner.api.validate_active_wheel(py)?;
        // Keep the Rc-owned native result untouched and transfer a fresh
        // copy-sharing handle. The private bridge consumes that handle on every
        // return path, including a null PyObject result.
        let transfer = value.copy_sharing()?;
        let raw = transfer.into_raw();
        let python_object = unsafe {
            (value.inner.api.eager_tensor_from_handle)(raw, false)
        };
        if python_object.is_null() {
            // Do not delete `raw`: EagerTensorFromHandle takes ownership even
            // when construction reports an error.
            return Err(python_error_or(
                py,
                "EagerTensorFromHandle returned null",
            ));
        }
        Ok(unsafe { Bound::from_owned_ptr(py, python_object) })
    }

    fn unary(input: &RxtTfTensor, op_name: &str) -> PyResult<RxtTfTensor> {
        let status = OwnedStatus::new(input.inner.api)?;
        let context = input.context();
        let device = input.backing_device()?;
        let op = OwnedOp::new(Rc::clone(&context), op_name, &status)?;
        op.set_device(&device, &status)?;
        op.add_input(input.pointer(), &status)?;
        op.set_type("T", TF_FLOAT)?;
        let result = RxtTfTensor::from_pending(op.execute_one(&status)?, context)?;
        result.validate(2)?;
        Ok(result)
    }

    fn binary(
        left: &RxtTfTensor,
        right: &RxtTfTensor,
        op_name: &str,
        matmul_attrs: bool,
        expected_rank: c_int,
    ) -> PyResult<RxtTfTensor> {
        same_context(left, right)?;
        let left_device = left.backing_device()?;
        let right_device = right.backing_device()?;
        if left_device != right_device {
            return Err(value_error(format!(
                "tensor device mismatch: {left_device} vs {right_device}"
            )));
        }
        let status = OwnedStatus::new(left.inner.api)?;
        let context = left.context();
        let op = OwnedOp::new(Rc::clone(&context), op_name, &status)?;
        op.set_device(&left_device, &status)?;
        op.add_input(left.pointer(), &status)?;
        op.add_input(right.pointer(), &status)?;
        op.set_type("T", TF_FLOAT)?;
        if matmul_attrs {
            op.set_bool("transpose_a", false)?;
            op.set_bool("transpose_b", false)?;
            op.set_bool("grad_a", false)?;
            op.set_bool("grad_b", false)?;
        }
        let result = RxtTfTensor::from_pending(op.execute_one(&status)?, context)?;
        result.validate(expected_rank)?;
        Ok(result)
    }

    pub fn matmul(left: &RxtTfTensor, right: &RxtTfTensor) -> PyResult<RxtTfTensor> {
        Python::attach(|_py| binary(left, right, "MatMul", true, 2))
    }

    pub fn add(left: &RxtTfTensor, right: &RxtTfTensor) -> PyResult<RxtTfTensor> {
        Python::attach(|_py| {
            let expected_rank = if left.rank()? == 2 || right.rank()? == 2 {
                2
            } else {
                1
            };
            binary(left, right, "AddV2", false, expected_rank)
        })
    }

    pub fn relu(input: &RxtTfTensor) -> PyResult<RxtTfTensor> {
        Python::attach(|_py| unary(input, "Relu"))
    }

    pub fn sigmoid(input: &RxtTfTensor) -> PyResult<RxtTfTensor> {
        Python::attach(|_py| unary(input, "Sigmoid"))
    }

    fn mean_axis_handle(
        context: Rc<BorrowedContext>,
    ) -> PyResult<Rc<OwnedTensorHandle>> {
        let api = context.api;
        let status = OwnedStatus::new(api)?;
        let tensor = OwnedTfTensor::axis_one(api)?;
        let raw = unsafe { (api.tfe_new_tensor_handle)(tensor.raw, status.pointer()) };
        let pending = PendingHandle::new(api, raw);
        status.check("TFE_NewTensorHandle(mean axis)")?;
        // TFE_NewTensorHandle retains the Tensor storage; the temporary
        // TF_Tensor remains caller-owned and is deleted immediately here.
        drop(tensor);
        pending.into_owned(context)
    }

    /// Reduce mean along the statically proven axis [1], keep_dims=false.
    pub fn reduce_mean_axis1(input: &RxtTfTensor) -> PyResult<RxtTfTensor> {
        Python::attach(|_py| {
            let status = OwnedStatus::new(input.inner.api)?;
            let context = input.context();
            let device = input.backing_device()?;
            let axis = mean_axis_handle(Rc::clone(&context))?;
            let op = OwnedOp::new(Rc::clone(&context), "Mean", &status)?;
            op.set_device(&device, &status)?;
            op.add_input(input.pointer(), &status)?;
            op.add_input(axis.raw, &status)?;
            op.set_type("T", TF_FLOAT)?;
            op.set_type("Tidx", TF_INT32)?;
            op.set_bool("keep_dims", false)?;
            let result =
                RxtTfTensor::from_pending(op.execute_one(&status)?, context)?;
            result.validate(1)?;
            Ok(result)
        })
    }

    #[allow(dead_code)]
    pub fn runtime_tensorflow_root() -> Option<&'static str> {
        API.get()
            .and_then(|api| api.tensorflow_root.to_str())
    }
}
"""


def runtime_module_helpers() -> str:
    """Return the exact generated TensorFlow runtime helper module."""
    return _RUNTIME_MODULE.strip() + "\n"


__all__ = ["runtime_module_helpers"]

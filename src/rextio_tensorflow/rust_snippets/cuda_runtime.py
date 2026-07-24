"""Generated Rust runtime for the bounded Linux TensorFlow CUDA E3 candidate.

The CPU helper remains byte-for-byte unchanged.  This module derives an
independent Rust module from that audited ownership/provenance foundation and
applies checked, single-occurrence transformations.  The generated module has
its own API cache and tensor type, so CPU and CUDA values cannot cross lanes.
"""

from __future__ import annotations

from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            f"rextio-tensorflow CUDA helper source drift at {label}: "
            f"expected one anchor, found {source.count(old)}"
        )
    return source.replace(old, new, 1)


def _build_cuda_runtime() -> str:
    source = runtime_module_helpers()
    source = source.replace("rextio_tensorflow_runtime", "rextio_tensorflow_cuda_runtime")
    source = source.replace("RxtTfTensor", "RxtTfCudaTensor")
    source = source.replace("extract_f32_cpu_2d", "extract_f32_cuda0_2d")
    source = source.replace("extract_f32_cpu_1d", "extract_f32_cuda0_1d")

    source = _replace_once(
        source,
        "use pyo3::types::PyAny;",
        "use pyo3::types::{PyAny, PyTuple};",
        "PyTuple import",
    )
    source = _replace_once(
        source,
        "    type TfTensor = c_void;\n",
        "    type TfTensor = c_void;\n    type TfDeviceList = c_void;\n",
        "TF_DeviceList type",
    )
    source = _replace_once(
        source,
        '    const EXPECTED_TF_VERSION: &str = "2.21.0";\n',
        """    const EXPECTED_TF_VERSION: &str = "2.21.0";
    const SYM_TAPE_POSSIBLE_GRADIENT_TYPES: &str =
        "_Z35TFE_Py_TapeSetPossibleGradientTypesP7_object";

    #[cfg(not(all(
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    )))]
    compile_error!(
        "rextio-tensorflow CUDA E3: build-only candidate supports only \
         x86_64-unknown-linux-gnu"
    );
""",
        "CUDA platform and tape symbol",
    )
    source = _replace_once(
        source,
        """        tf_tensor_byte_size: unsafe extern "C" fn(*const TfTensor) -> usize,

        tfe_new_tensor_handle:""",
        """        tf_tensor_byte_size: unsafe extern "C" fn(*const TfTensor) -> usize;
        tf_delete_device_list: unsafe extern "C" fn(*mut TfDeviceList),
        tf_device_list_count: unsafe extern "C" fn(*const TfDeviceList) -> c_int,
        tf_device_list_name: unsafe extern "C" fn(
            *const TfDeviceList,
            c_int,
            *mut TfStatus,
        ) -> *const c_char,
        tf_device_list_type: unsafe extern "C" fn(
            *const TfDeviceList,
            c_int,
            *mut TfStatus,
        ) -> *const c_char,

        tfe_context_list_devices:
            unsafe extern "C" fn(*mut TfeContext, *mut TfStatus) -> *mut TfDeviceList,
        tfe_new_tensor_handle:""",
        "device-list API fields",
    )
    source = _replace_once(
        source,
        """        eager_tensor_check_exact:
            unsafe extern "C" fn(*const pyo3::ffi::PyObject) -> bool,
""",
        """        eager_tensor_check_exact:
            unsafe extern "C" fn(*const pyo3::ffi::PyObject) -> bool,
        tfe_py_tape_set_possible_gradient_types:
            unsafe extern "C" fn(
                *mut pyo3::ffi::PyObject,
            ) -> *mut pyo3::ffi::PyObject,
""",
        "tape API field",
    )
    source = _replace_once(
        source,
        """                    tf_tensor_byte_size: framework.resolve("TF_TensorByteSize")?,

                    tfe_new_tensor_handle:""",
        """                    tf_tensor_byte_size: framework.resolve("TF_TensorByteSize")?,
                    tf_delete_device_list: framework.resolve("TF_DeleteDeviceList")?,
                    tf_device_list_count: framework.resolve("TF_DeviceListCount")?,
                    tf_device_list_name: framework.resolve("TF_DeviceListName")?,
                    tf_device_list_type: framework.resolve("TF_DeviceListType")?,

                    tfe_context_list_devices: cc.resolve("TFE_ContextListDevices")?,
                    tfe_new_tensor_handle:""",
        "device-list symbol resolution",
    )
    source = _replace_once(
        source,
        """                    eager_tensor_check_exact: pywrap.resolve(SYM_EAGER_TENSOR_CHECK_EXACT)?,
""",
        """                    eager_tensor_check_exact: pywrap.resolve(SYM_EAGER_TENSOR_CHECK_EXACT)?,
                    tfe_py_tape_set_possible_gradient_types:
                        pywrap.resolve(SYM_TAPE_POSSIBLE_GRADIENT_TYPES)?,
""",
        "tape symbol resolution",
    )

    device_list_owner = r"""
    struct OwnedDeviceList {
        api: &'static Api,
        raw: *mut TfDeviceList,
    }

    impl Drop for OwnedDeviceList {
        fn drop(&mut self) {
            if !self.raw.is_null() {
                unsafe { (self.api.tf_delete_device_list)(self.raw) };
                self.raw = std::ptr::null_mut();
            }
        }
    }

"""
    source = _replace_once(
        source,
        "    struct BorrowedContext {\n",
        device_list_owner + "    struct BorrowedContext {\n",
        "device-list RAII",
    )
    source = _replace_once(
        source,
        """            Ok(Rc::new(Self {
                api,
                raw: raw.cast::<TfeContext>(),
                _python_context: context.unbind(),
                _python_capsule: capsule.unbind(),
                _thread_affine: PhantomData,
            }))
        }
    }
""",
        """            Ok(Rc::new(Self {
                api,
                raw: raw.cast::<TfeContext>(),
                _python_context: context.unbind(),
                _python_capsule: capsule.unbind(),
                _thread_affine: PhantomData,
            }))
        }

        fn exact_gpu0_device(&self) -> PyResult<String> {
            let status = OwnedStatus::new(self.api)?;
            let raw = unsafe {
                (self.api.tfe_context_list_devices)(self.raw, status.pointer())
            };
            let devices = OwnedDeviceList { api: self.api, raw };
            status.check("TFE_ContextListDevices")?;
            if devices.raw.is_null() {
                return Err(runtime_error("TFE_ContextListDevices returned null"));
            }
            let count = unsafe {
                (self.api.tf_device_list_count)(devices.raw.cast_const())
            };
            if count < 0 {
                return Err(runtime_error("TF_DeviceListCount returned a negative count"));
            }
            let mut matches = Vec::new();
            for index in 0..count {
                let type_pointer = unsafe {
                    (self.api.tf_device_list_type)(
                        devices.raw.cast_const(),
                        index,
                        status.pointer(),
                    )
                };
                status.check("TF_DeviceListType")?;
                let name_pointer = unsafe {
                    (self.api.tf_device_list_name)(
                        devices.raw.cast_const(),
                        index,
                        status.pointer(),
                    )
                };
                status.check("TF_DeviceListName")?;
                if type_pointer.is_null() || name_pointer.is_null() {
                    return Err(runtime_error("TensorFlow device list returned a null field"));
                }
                let device_type = unsafe { CStr::from_ptr(type_pointer) }.to_string_lossy();
                let name = unsafe { CStr::from_ptr(name_pointer) }
                    .to_string_lossy()
                    .into_owned();
                if device_type == "GPU" && name.ends_with("/device:GPU:0") {
                    matches.push(name);
                }
            }
            if matches.len() != 1 {
                return Err(value_error(format!(
                    "expected exactly one TensorFlow GPU:0 device, found {}",
                    matches.len()
                )));
            }
            Ok(matches.remove(0))
        }
    }
""",
        "borrowed context GPU enumeration",
    )

    old_backing = """        fn backing_device(&self) -> PyResult<String> {
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
"""
    new_backing = """        fn backing_device(&self) -> PyResult<String> {
            let expected = self.inner.context.exact_gpu0_device()?;
            let status = OwnedStatus::new(self.inner.api)?;
            let pointer = unsafe {
                (self.inner.api.tfe_tensor_handle_backing_device_name)(
                    self.inner.raw,
                    status.pointer(),
                )
            };
            status.check("TFE_TensorHandleBackingDeviceName")?;
            if pointer.is_null() {
                return Err(value_error("expected a TensorFlow GPU:0 tensor"));
            }
            let actual = unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned();
            if actual != expected {
                return Err(value_error(format!(
                    "expected exact GPU:0 device {expected}, got {actual}"
                )));
            }
            Ok(actual)
        }
"""
    source = _replace_once(source, old_backing, new_backing, "exact GPU backing device")

    tape_guard = r"""
    fn reject_gradient_recording(
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
        api: &'static Api,
    ) -> PyResult<()> {
        let tensors = PyTuple::new(py, [value])?;
        let raw = unsafe {
            (api.tfe_py_tape_set_possible_gradient_types)(tensors.as_ptr())
        };
        if raw.is_null() {
            return Err(python_error_or(
                py,
                "TFE_Py_TapeSetPossibleGradientTypes returned null",
            ));
        }
        let result = unsafe { Bound::from_owned_ptr(py, raw) };
        let possible: i64 = result.extract()?;
        if possible != 0 {
            return Err(runtime_error(format!(
                "CUDA E3 requires no active backward tape or forward accumulator; \
                 possible gradient type was {possible}"
            )));
        }
        Ok(())
    }

"""
    source = _replace_once(
        source,
        "    fn extract_common(\n",
        tape_guard + "    fn extract_common(\n",
        "gradient guard helper",
    )
    source = _replace_once(
        source,
        """        let api = load_api(py)?;
        let context = BorrowedContext::from_python(py, api)?;
        let object = value.as_ptr().cast_const();
""",
        """        let api = load_api(py)?;
        let context = BorrowedContext::from_python(py, api)?;
        reject_gradient_recording(py, value, api)?;
        let object = value.as_ptr().cast_const();
""",
        "gradient guard call",
    )
    materializers = r"""
    pub fn materialize_f32_cuda0_2d(
        py: Python<'_>,
        value: RxtTfCudaTensor,
    ) -> PyResult<Bound<'_, PyAny>> {
        value.validate_f32(2)?;
        materialize_tensor(py, value)
    }

    pub fn materialize_f32_cuda0_1d(
        py: Python<'_>,
        value: RxtTfCudaTensor,
    ) -> PyResult<Bound<'_, PyAny>> {
        value.validate_f32(1)?;
        materialize_tensor(py, value)
    }

"""
    source = _replace_once(
        source,
        "    fn unary(input: &RxtTfCudaTensor, op_name: &str) -> PyResult<RxtTfCudaTensor> {\n",
        materializers
        + "    fn unary(input: &RxtTfCudaTensor, op_name: &str) -> PyResult<RxtTfCudaTensor> {\n",
        "rank-specific materializers",
    )
    return source


_CUDA_RUNTIME_MODULE = _build_cuda_runtime()


def cuda_runtime_module_helpers() -> str:
    """Return the separate exact-text generated CUDA runtime module."""
    return _CUDA_RUNTIME_MODULE


__all__ = ["cuda_runtime_module_helpers"]

"""Source-contract tests for invocation-local trusted resident fact reuse.

These tests do not execute TFE. They prove the generated runtime encodes the
safe 0.1.3 boundary:

* Core plugin API 1.6 has no per-generated-function invocation/prologue hook.
* Reuse is value-carried (``TrustedResidentFacts`` on ``OwnedTensorHandle``)
  plus the existing context-bound prepared-constant table — never a process-
  global, thread-local, or cross-context tensor/context cache.
* Boundary tensors are still validated fully once before facts are stamped;
  operation results must be validated before their facts are trusted.
* ``TF_Status`` remains per-operation because no public ``TF_ResetStatus``
  exists on the TensorFlow 2.21.0 wheel images this plugin binds.
"""

from __future__ import annotations

from rextio_tensorflow.rust_snippets.cuda_runtime import cuda_runtime_module_helpers
from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers


def test_trusted_resident_facts_are_value_carried_not_global() -> None:
    helper = runtime_module_helpers()
    assert "struct TrustedResidentFacts" in helper
    assert "facts: RefCell<Option<TrustedResidentFacts>>" in helper
    assert "facts: RefCell::new(None)" in helper
    # No process-global / thread-local tensor or facts tables.
    assert "static FACTS" not in helper
    assert "OnceLock<TrustedResidentFacts>" not in helper
    assert "thread_local!" not in helper
    assert "lazy_static" not in helper
    assert "process-global or thread-local tensor/context cache" in helper


def test_core_lacks_invocation_prologue_hook_is_documented() -> None:
    helper = runtime_module_helpers()
    assert "no per-generated-function invocation/prologue" in helper
    assert "true explicit InvocationContext" in helper
    assert "Core prologue hook" in helper


def test_boundary_validation_stamps_facts_after_full_checks() -> None:
    helper = runtime_module_helpers()
    validate = helper[
        helper.index("fn validate_typed(&self, expected_type: c_int, expected_rank: c_int)") : helper.index(
            "fn validate_f32(&self, expected_rank: c_int)"
        )
    ]
    # Full fail-closed path still performs the TFE queries once (dtype here;
    # rank/device via helpers that hit TFE when facts are absent).
    assert "tfe_tensor_handle_data_type" in validate
    assert "let rank = self.rank()?" in validate
    assert "TFE_TensorHandleDim" in validate or "tfe_tensor_handle_dim" in validate
    assert "let device = self.backing_device()?" in validate
    assert "tfe_tensor_handle_num_dims" in helper
    assert "TFE_TensorHandleBackingDeviceName" in helper
    # Facts are stamped only after those checks succeed.
    assert "TrustedResidentFacts {" in validate
    assert "dtype: actual_type" in validate
    # Trusted reuse short-circuits only when dtype+rank facts already match.
    assert "facts.dtype == expected_type && facts.rank == expected_rank" in validate
    assert "avoid redundant dtype/rank/device" in validate
    # Mismatched facts are cleared before re-validation (fail-closed recovery).
    assert "*self.inner.facts.borrow_mut() = None;" in validate


def test_rank_and_device_reuse_trusted_facts() -> None:
    helper = runtime_module_helpers()
    assert "Trusted resident intermediate: reuse the validated rank fact" in helper
    assert "Trusted resident facts already include a validated device string" in helper
    # Boundary extract still validates exactly once before return.
    extract = helper[
        helper.index("fn extract_common(") : helper.index("pub fn extract_f32_cpu_2d(")
    ]
    assert "tensor.validate_f32(expected_rank)?" in extract or (
        "validate_f32(expected_rank)" in extract
    )
    assert "validate_i64(expected_rank)" in extract


def test_operation_results_validated_before_facts_trusted() -> None:
    helper = runtime_module_helpers()
    # Unary/binary/reductions still validate results after TFE_Execute.
    assert "result.validate_f32(expected_rank)?" in helper
    assert "result.validate_i64(1)?" in helper
    # Inputs of untrusted handles still go through validate paths.
    assert "input.validate_f32(expected_rank)?" in helper
    assert "input.validate_f32(2)?" in helper


def test_status_allocation_remains_per_operation_without_reset_symbol() -> None:
    helper = runtime_module_helpers()
    assert "TF_Status allocation remains per operation" in helper
    assert "no exact public" in helper
    assert "TF_ResetStatus" in helper
    # Only allocate/delete symbols are resolved from the framework image.
    assert 'framework.resolve("TF_NewStatus")' in helper
    assert 'framework.resolve("TF_DeleteStatus")' in helper
    assert 'framework.resolve("TF_ResetStatus")' not in helper
    assert "OwnedStatus::new(" in helper


def test_drop_ordering_and_raii_contracts_preserved() -> None:
    helper = runtime_module_helpers()
    assert "impl Drop for BorrowedContext" in helper
    assert "prepared.replace(PreparedConstantTable::empty())" in helper
    assert "impl Drop for OwnedTensorHandle" in helper
    assert "impl Drop for PendingHandle" in helper
    assert "impl Drop for PreparedHandle" in helper
    assert "impl Drop for OwnedStatus" in helper
    struct_body = helper[
        helper.index("struct BorrowedContext {") : helper.index(
            "impl Drop for BorrowedContext"
        )
    ]
    assert struct_body.index("_python_context") < struct_body.index("prepared:")
    assert struct_body.index("_python_capsule") < struct_body.index("prepared:")
    # Owned handle still keeps context Rc so anchors outlive the handle.
    owned = helper[
        helper.index("struct OwnedTensorHandle {") : helper.index(
            "impl Drop for OwnedTensorHandle"
        )
    ]
    assert "context: Rc<BorrowedContext>" in owned
    assert "facts: RefCell<Option<TrustedResidentFacts>>" in owned


def test_no_cross_context_fact_sharing_construct() -> None:
    helper = runtime_module_helpers()
    # Facts live on the handle, not on BorrowedContext; prepared constants remain
    # the only context-table reuse and stay isolated per BorrowedContext value.
    assert "prepared: RefCell<PreparedConstantTable>" in helper
    assert "Distinct BorrowedContext values never share slots" in helper
    # same_context still compares raw TFE_Context*, not fact tables.
    assert "tensor operands do not belong to the same Python eager context" in helper


def test_cuda_derivation_preserves_trusted_facts_and_gpu_device_path() -> None:
    cuda = cuda_runtime_module_helpers()
    assert "struct TrustedResidentFacts" in cuda
    assert "facts: RefCell<Option<TrustedResidentFacts>>" in cuda
    assert "Trusted resident facts already include a validated device string" in cuda
    assert "exact_gpu0_device" in cuda
    assert "expected a TensorFlow GPU:0 tensor" in cuda
    assert "expected a CPU tensor" not in cuda[
        cuda.index("fn backing_device(&self)") : cuda.index("fn rank(&self)")
    ]
    assert "TF_Status allocation remains per operation" in cuda
    assert "thread_local!" not in cuda
    assert "OnceLock<TrustedResidentFacts>" not in cuda
    # CUDA E3 surface remains reduced.
    assert "pub fn transpose(" not in cuda
    assert "pub fn extract_i64_cpu_1d" not in cuda

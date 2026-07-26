"""Source-contract tests for context-bound prepared constant handles.

These tests do not execute TFE. They prove the generated runtime encodes the
safe 0.1.3 residency boundary: prepared axes/permutations are owned by a
``BorrowedContext`` RAII table, never by a process-global cache, and cannot be
reused across distinct context values.
"""

from __future__ import annotations

from rextio_tensorflow.rust_snippets.cuda_runtime import cuda_runtime_module_helpers
from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers


def test_prepared_constants_are_context_bound_not_process_global() -> None:
    helper = runtime_module_helpers()
    assert "struct PreparedConstantTable" in helper
    assert "struct PreparedHandle" in helper
    assert "prepared: RefCell<PreparedConstantTable>" in helper
    assert "PreparedConstantTable::empty()" in helper
    # API symbol table may remain OnceLock; prepared constants must not.
    assert "static PREPARED" not in helper
    assert "OnceLock<PreparedConstantTable>" not in helper
    assert "lazy_static" not in helper
    assert "thread_local!" not in helper
    assert "Handles are never process-global" in helper
    assert "cross-generated-function residency" in helper


def test_prepared_constant_slots_cover_axes_and_transpose_perm() -> None:
    helper = runtime_module_helpers()
    for field in (
        "reduction_axis_0",
        "reduction_axis_1",
        "argmax_axis_0",
        "argmax_axis_1",
        "transpose_perm_rank2",
    ):
        assert field in helper
    assert "prepared_reduction_axis" in helper
    assert "prepared_argmax_axis" in helper
    assert "prepared_transpose_perm_rank2" in helper
    assert "make_prepared_reduction_axis" in helper
    assert "make_prepared_argmax_axis" in helper
    assert "make_prepared_transpose_perm_rank2" in helper


def test_prepared_handle_raii_deletes_owned_tensor_handles() -> None:
    helper = runtime_module_helpers()
    assert "impl Drop for PreparedHandle" in helper
    assert "tfe_delete_tensor_handle" in helper
    # Ownership stays with the context table; ops borrow the raw pointer.
    assert "fn pointer(&self) -> *mut TfeTensorHandle" in helper
    assert "context.prepared_reduction_axis(axis)" in helper
    assert "context.prepared_argmax_axis(axis)" in helper
    assert "context.prepared_transpose_perm_rank2()" in helper


def test_borrowed_context_drop_clears_prepared_handles_before_python_anchors() -> None:
    """Source contract: prepared handles drop while Python anchors still live."""
    helper = runtime_module_helpers()
    assert "impl Drop for BorrowedContext" in helper
    assert "prepared.replace(PreparedConstantTable::empty())" in helper
    drop_impl = helper[
        helper.index("impl Drop for BorrowedContext") : helper.index(
            "impl BorrowedContext {"
        )
    ]
    assert "prepared.replace(PreparedConstantTable::empty())" in drop_impl
    assert "_python_context" in drop_impl or "Python anchors" in drop_impl or (
        "python" in drop_impl.lower()
    )
    # Comment must document declaration-order hazard explicitly.
    assert "declaration order" in drop_impl or "Declaration-order" in drop_impl
    assert "while" in drop_impl and "alive" in drop_impl
    # Field layout still declares prepared after the Python anchors so default
    # field drop would be unsafe without the explicit Drop above.
    struct_body = helper[
        helper.index("struct BorrowedContext {") : helper.index(
            "impl Drop for BorrowedContext"
        )
    ]
    assert struct_body.index("_python_context") < struct_body.index("prepared:")
    assert struct_body.index("_python_capsule") < struct_body.index("prepared:")


def test_transpose_default_perm_is_rank2_swap() -> None:
    helper = runtime_module_helpers()
    assert "fn transpose_perm_rank2" in helper
    assert "pub fn transpose(" in helper
    assert '"Transpose"' in helper
    assert '"Tperm"' in helper
    # Default Python rank-2 perm writes [1, 0] into the int32 vector.
    assert "std::ptr::write(values, 1);" in helper
    assert "std::ptr::write(values.add(1), 0);" in helper


def test_context_isolation_documented_for_distinct_borrowed_contexts() -> None:
    helper = runtime_module_helpers()
    assert "Distinct BorrowedContext values never share slots" in helper
    assert "same raw `TFE_Context*`" in helper
    assert "Graph / FunctionDef fusion" in helper


def test_cuda_derivation_preserves_context_bound_prepared_table() -> None:
    cuda = cuda_runtime_module_helpers()
    assert "struct PreparedConstantTable" in cuda
    assert "prepared: RefCell<PreparedConstantTable>" in cuda
    assert "prepared_reduction_axis" in cuda
    assert "make_prepared_reduction_axis" in cuda
    assert "pub fn reduce_mean_axis1" in cuda
    # CUDA inherits the same Drop-order safety for prepared handles.
    assert "impl Drop for BorrowedContext" in cuda
    assert "prepared.replace(PreparedConstantTable::empty())" in cuda
    # CUDA E3 surface still excludes default transpose public entry.
    assert "pub fn transpose(" not in cuda
    assert "OnceLock<PreparedConstantTable>" not in cuda

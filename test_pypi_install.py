"""
ezlmm 0.4.3 smoke test — run after installing from TestPyPI.

Usage:
    # Install from TestPyPI first:
    pip install -i https://test.pypi.org/simple/ ezlmm

    # Then run this script:
    python test_pypi_install.py
"""

import sys

RPY2_AVAILABLE = False
try:
    import rpy2
    RPY2_AVAILABLE = True
except ImportError:
    pass


def test_version():
    """Test that __version__ is correct."""
    import ezlmm
    assert ezlmm.__version__ == "0.4.7", f"Expected 0.4.7, got {ezlmm.__version__}"
    print(f"  PASS: version is {ezlmm.__version__}")
    return True


def test_no_core_py():
    """Verify ezlmm.core does NOT exist (backward compat removed)."""
    import ezlmm
    assert not hasattr(ezlmm, 'core'), "ezlmm.core should not exist"
    print("  PASS: no ezlmm.core (backward compat removed)")
    return True


def test_report_functions():
    """Verify report functions exist and have correct signatures."""
    from ezlmm.report import write_simple_effect_lmm, write_simple_effect_glmm
    import inspect
    for func in (write_simple_effect_lmm, write_simple_effect_glmm):
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        expected = ["dep_var", "trans_dict", "optimal_model", "df_anova", "robj_model", "data"]
        assert params == expected, f"{func.__name__}: expected {expected}, got {params}"
        print(f"  PASS: {func.__name__}{sig}")
    return True


def test_extract_contrast():
    """Test extract_contrast function."""
    from ezlmm.report import extract_contrast
    # Minimal test — parse a known contrast string
    fake = """
key1
value1
key2
value2
"""
    result = extract_contrast(fake.strip(), factor_num=1)
    assert isinstance(result, list), "Should return a list"
    print(f"  PASS: extract_contrast returns {len(result)} contrast(s)")
    return True


def test_lazy_loading():
    """Test that __getattr__ lazy loading works for non-R names."""
    import ezlmm
    names_to_test = ["extract_contrast", "r_object", "r2p", "p2r"]
    for name in names_to_test:
        obj = getattr(ezlmm, name)
        assert obj is not None, f"ezlmm.{name} is None"
        print(f"    {name}: OK")
    print("  PASS: lazy loading OK for non-R names")
    return True


def test_submodules():
    """Test that non-R submodules are accessible."""
    from ezlmm.report import extract_contrast
    from ezlmm.utils import r_object, r2p, p2r
    print("  PASS: non-R submodules accessible")
    return True


def test_model_classes_require_rpy2():
    """Model classes require rpy2 — this is expected behavior."""
    if not RPY2_AVAILABLE:
        print("  SKIP: rpy2 not installed (needed for model classes)")
        return True
    # If rpy2 IS available, test that model classes can be imported
    from ezlmm import LinearMixedModel, GeneralizedLinearMixedModel
    print("  PASS: model classes import OK (rpy2 available)")
    return True


def main():
    tests = [
        ("Version test", test_version),
        ("No core.py test", test_no_core_py),
        ("Report function signatures", test_report_functions),
        ("extract_contrast basic test", test_extract_contrast),
        ("Lazy loading (non-R names)", test_lazy_loading),
        ("Non-R submodules", test_submodules),
        ("Model classes (rpy2 dependent)", test_model_classes_require_rpy2),
    ]

    print("=" * 50)
    print("ezlmm 0.4.7 Smoke Test")
    print(f"rpy2 available: {RPY2_AVAILABLE}")
    print("=" * 50)

    all_passed = True
    for name, fn in tests:
        print(f"\n{name}...")
        try:
            if not fn():
                all_passed = False
                print(f"  FAIL")
        except Exception as e:
            all_passed = False
            print(f"  ERROR: {e}")

    print("\n" + "=" * 50)
    if all_passed:
        print("All tests PASSED")
        return 0
    else:
        print("Some tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())

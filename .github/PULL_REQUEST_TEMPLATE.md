## Description

<!-- A clear description of what this PR does and the problem it solves. -->

## Related Issue

<!-- Link to the issue this PR addresses (e.g., Closes #123). -->

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to change)
- [ ] Documentation update
- [ ] UI / Frontend change (changes to the local dashboard)
- [ ] Refactoring (no functional changes)

## Visual Changes (if applicable)

<!-- If this PR changes the local dashboard UI or documentation, include before and after screenshots. -->
<details>
<summary>Click to expand screenshots</summary>

| Before | After |
| :--- | :--- |
| <!-- screenshot --> | <!-- screenshot --> |

</details>

## Verification & Manual Testing

<!-- REQUIRED: You must provide proof of testing before your PR can be merged. -->

### For Core Library Changes (`beliefstate/`)
1. Use `test_package/` to write and run a verification script
2. Upload BEFORE screenshot showing the bug/old behavior
3. Upload AFTER screenshot showing the fix/new feature working

### For Documentation and Dashboard Fixes
1. Upload BEFORE screenshot showing the issue
2. Upload AFTER screenshot showing your fix

### Proof of Testing
- [ ] I have verified my changes and provided before-and-after visual proof
- **Verification method**: <!-- e.g. test_package/test_fix.py, manual dashboard testing -->
- **BEFORE screenshot**:
<!-- DRAG & DROP BEFORE SCREENSHOT HERE -->
- **AFTER screenshot**:
<!-- DRAG & DROP AFTER SCREENSHOT HERE -->

## Checklist

- [ ] I have read the [CONTRIBUTING](https://github.com/AltioraLabs/beliefstate/blob/main/CONTRIBUTING.md) guide
- [ ] My code follows the project's coding standards (`ruff check .` passes)
- [ ] My code is formatted correctly (`ruff format --check .` passes)
- [ ] My type checks pass (`mypy beliefstate` passes)
- [ ] I have added tests that prove my fix is effective or my feature works
- [ ] All new and existing tests pass (`pytest`)
- [ ] I have updated documentation if needed
- [ ] I have added an entry to `CHANGELOG.md` under `[Unreleased]`

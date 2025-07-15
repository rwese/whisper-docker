# TODO Items

## Re-enable Development Tooling

### Context
The Python linting and formatting tools (black, isort, flake8, pylint, mypy, pytest) were temporarily disabled in `.pre-commit-config.yaml` to prevent AI agent context disruption during development.

### Re-enabling Instructions

1. **Uncomment the disabled hooks** in `.pre-commit-config.yaml`:
   - Remove the `#` comments from lines 20-87
   - This will re-enable: black, isort, flake8, pylint, mypy, pytest

2. **Fix outstanding issues** before re-enabling:
   - **Flake8**: Fix undefined variable issue in web_service.py line 369
   - **Mypy**: Fix Optional[str] type annotation in get_error_response function
   - **Mypy**: Fix temperature validation type checking (line 816)
   - **Pylint**: Address code quality warnings (too many lines, etc.)
   - **Pytest**: Fix failing test cases in test_async_api.py

3. **Test the re-enabled hooks**:
   ```bash
   pre-commit run --all-files
   ```

4. **Alternative: Manual execution**:
   ```bash
   make format  # Run black and isort
   make lint    # Run flake8 and pylint
   make test    # Run pytest
   ```

### Priority Issues to Address

1. **High Priority**:
   - Fix flake8 F821 error (undefined name 'e')
   - Fix mypy type annotation errors
   - Fix failing pytest tests

2. **Medium Priority**:
   - Address pylint code quality warnings
   - Refactor long modules (web_service.py > 1000 lines)

3. **Low Priority**:
   - Clean up duplicate code warnings
   - Optimize lambda usage

### Benefits of Re-enabling

- **Code Quality**: Consistent formatting and style enforcement
- **Error Prevention**: Early detection of type errors and bugs
- **Test Coverage**: Automated test execution on commits
- **Team Collaboration**: Standardized code style for contributors

---

## Other TODO Items

Add other development tasks here as needed.

---
name: Issue #1
about: Fix empty code snippet handling
title: 'Bug: Empty code snippet handling'
labels: bug
assignees: ''

---

**Describe the bug**
The `process_code` function in `core.py` doesn't handle empty code snippets correctly. It should return a proper error message when an empty string is passed, but it's not working as expected.

**To Reproduce**
Call the `process_code` function with an empty string:
```python
from tom_swe.core import process_code
result = process_code("")
print(result)
```

**Expected behavior**
The function should return a dictionary with an error message indicating that the code snippet cannot be empty.

**Additional context**
There's also a missing test case for this scenario in the test suite.
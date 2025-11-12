# Release Process for tom-swe

This document describes how to publish new versions of `tom-swe` to PyPI.

## Prerequisites

### 1. Configure PyPI Trusted Publishing

Trusted Publishing is the modern, secure way to publish to PyPI without managing API tokens. It uses OpenID Connect (OIDC) to authenticate GitHub Actions directly with PyPI.

#### Steps to set up on PyPI:

1. **Create a PyPI account** (if you don't have one):
   - Go to https://pypi.org/account/register/
   - Verify your email

2. **Set up Trusted Publishing**:
   - Go to https://pypi.org/manage/account/publishing/
   - Click "Add a new pending publisher"
   - Fill in the form:
     - **PyPI Project Name**: `tom-swe`
     - **Owner**: `All-Hands-AI`
     - **Repository name**: `ToM-SWE`
     - **Workflow name**: `publish-to-pypi.yml`
     - **Environment name**: `pypi`
   - Click "Add"

3. **Configure GitHub Environment** (optional but recommended):
   - Go to your GitHub repository settings
   - Navigate to "Environments"
   - Create a new environment named `pypi`
   - Add protection rules (e.g., require reviewers for production releases)

### 2. Test with TestPyPI (Optional but Recommended)

Before publishing to the main PyPI, you can test with TestPyPI:

1. **Set up TestPyPI Trusted Publishing**:
   - Go to https://test.pypi.org/manage/account/publishing/
   - Follow the same steps as above, but use environment name `testpypi`

2. **Manually trigger test publish**:
   - Go to Actions tab in GitHub
   - Select "Publish to PyPI" workflow
   - Click "Run workflow"
   - Check "Publish to TestPyPI instead of PyPI"
   - Click "Run workflow"

3. **Test installation**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tom-swe
   ```

## Release Process

### 1. Update Version Number

Edit `pyproject.toml` and update the version:

```toml
[project]
name = "tom-swe"
version = "1.0.1"  # Update this
```

**Version numbering convention** (Semantic Versioning):
- `MAJOR.MINOR.PATCH` (e.g., `1.0.1`)
- **MAJOR**: Breaking changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### 2. Update CHANGELOG (if you have one)

Document what changed in this release.

### 3. Commit and Push

```bash
git add pyproject.toml
git commit -m "Bump version to 1.0.1"
git push origin main
```

### 4. Create a Git Tag

```bash
git tag v1.0.1
git push origin v1.0.1
```

### 5. Create a GitHub Release

1. Go to https://github.com/All-Hands-AI/ToM-SWE/releases/new
2. Choose the tag you just created (`v1.0.1`)
3. Set the release title (e.g., `v1.0.1`)
4. Write release notes describing:
   - New features
   - Bug fixes
   - Breaking changes
   - Any migration steps needed
5. Click "Publish release"

### 6. Automated Publishing

Once you publish the GitHub release:
- The `publish-to-pypi.yml` workflow will automatically trigger
- It will build the package
- It will publish to PyPI using trusted publishing
- You can monitor progress in the Actions tab

### 7. Verify Publication

Check that your package is available:
- Visit https://pypi.org/project/tom-swe/
- Install it: `pip install tom-swe`

## Troubleshooting

### "Publisher already exists" error on PyPI

If you get this error when setting up trusted publishing:
- The package might already exist on PyPI
- You may need to claim ownership or contact PyPI support
- For first-time publishing, use the "pending publisher" feature

### Workflow fails with "Trusted publishing exchange failure"

1. Verify the environment name in the workflow matches what you set up on PyPI
2. Check that the repository name and owner are correct
3. Ensure the workflow file name is exactly `publish-to-pypi.yml`

### Build fails

```bash
# Test the build locally
python -m pip install build
python -m build

# Check the dist/ directory
ls -la dist/
```

### Version conflicts

If PyPI rejects because the version already exists:
- You cannot re-upload the same version
- Increment the version number (even for fixes)
- Consider using version suffixes for testing: `1.0.1rc1`, `1.0.1a1`

## Alternative: Manual Publishing (Not Recommended)

If you need to publish manually:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Upload to PyPI (requires API token)
twine upload dist/*
```

However, **always prefer the automated workflow** as it's more secure and consistent.

## Version Management Best Practices

1. **Never reuse version numbers** - Each release must have a unique version
2. **Follow semantic versioning** - Makes it clear what changed
3. **Tag all releases in git** - Makes it easy to track what was released
4. **Write clear release notes** - Helps users understand what changed
5. **Test with TestPyPI first** - Catch issues before production release

## For Users Installing from GitHub (Not Recommended)

If users install directly from GitHub:
```bash
pip install git+https://github.com/All-Hands-AI/ToM-SWE.git
```

This causes issues with:
- Binary builds (PyInstaller, cx_Freeze)
- Reproducible builds
- Dependency resolution
- Version pinning

**Always recommend installing from PyPI instead:**
```bash
pip install tom-swe
```

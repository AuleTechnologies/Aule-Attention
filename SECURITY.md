# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in aule-attention, please report it responsibly:

1. **Do NOT open a public GitHub issue** for security vulnerabilities
2. **Email:** Send details to security@aule.dev (or contact@aule.dev if unavailable)
3. **Include:**
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### What to Expect

- **Acknowledgment:** Within 48 hours
- **Initial Assessment:** Within 7 days
- **Resolution Timeline:** Depends on severity
  - Critical: 24-72 hours
  - High: 7 days
  - Medium: 30 days
  - Low: Next release cycle

## Security Practices

### No Pre-built Binaries in Git

This project does not ship pre-built binaries (`.dll`, `.so`, `.dylib`, `.spv`) in the git repository. This prevents supply chain attacks where malicious code could be injected into binary files.

**Users should:**
- Build from source using `zig build -Doptimize=ReleaseFast`
- Or download official releases from GitHub Releases with verified checksums

**If you find a binary file in this repository, please report it immediately.**

### Code Review

All changes go through code review before merging. External contributions require:
- Signed commits (recommended)
- Clear description of changes
- No binary files

### Dependency Management

- Dependencies are pinned with upper bounds to prevent unexpected breaking changes
- We regularly audit dependencies for known vulnerabilities

## Security Checklist for Contributors

Before submitting a PR:

- [ ] No hardcoded secrets, API keys, or credentials
- [ ] No binary files (`.dll`, `.so`, `.dylib`, `.exe`, `.spv`)
- [ ] No `eval()`, `exec()`, or similar dynamic code execution
- [ ] No pickle/marshal deserialization of untrusted data
- [ ] Input validation for any user-provided data
- [ ] Dependencies added are from trusted sources

## Known Security Considerations

### Native Library Loading

The Vulkan backend uses `ctypes.CDLL` to load the native library. The library path is determined at package install time from a known location within the package directory, not from user input.

### GPU Memory

GPU buffers are allocated and managed by the Vulkan runtime. The library does not expose raw pointers to Python code beyond what's necessary for the ctypes interface.

### Shell Scripts

Scripts in `scripts/` are provided for convenience but should be reviewed before execution, as they may:
- Add system repositories
- Install packages
- Require root privileges

Always review shell scripts before running them.

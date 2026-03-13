# Contributing to SO-101 Robotic Arm RL

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Your environment (Python version, OS, etc.)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:
- A clear description of the enhancement
- Why it would be useful
- Any implementation ideas you have

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** with clear, descriptive commits
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Submit a pull request**

#### Pull Request Guidelines

- Follow the existing code style
- Write clear commit messages
- Include tests if applicable
- Update README.md if adding new features
- Keep PRs focused on a single feature/fix

## Development Setup

1. Clone your fork:
```bash
git clone https://github.com/yourusername/SO101-Robotic-Arm-RL.git
cd SO101-Robotic-Arm-RL
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -r requirements.txt
pip install -e .
```

4. Create a branch:
```bash
git checkout -b feature/your-feature-name
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to classes and functions
- Keep functions focused and concise

## Testing

Before submitting a PR:
- Test your changes locally
- Run training for at least 1000 steps to verify no errors
- Test with multiple random seeds if applicable

## Adding New Training Stages

When adding a new training stage:

1. Create a new environment file: `envs/stage_X_task.py`
2. Inherit from `SO101BaseEnv`
3. Implement the reward function
4. Add clear documentation in the docstring
5. Update `ENV_MAP` in `scripts/train.py` and `scripts/evaluate.py`
6. Add documentation in `docs/`

## Documentation

- Update README.md for new features
- Add detailed docstrings
- Create documentation in `docs/` for complex features
- Include examples in docstrings

## Questions?

Feel free to open an issue for any questions about contributing!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

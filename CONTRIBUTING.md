# Contributing to RayWhisper2

Thank you for your interest in contributing to RayWhisper2! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/RayWhisper2.git
cd RayWhisper2
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### 4. Install Pre-commit Hooks (Optional)

```bash
pre-commit install
```

## Architecture Principles

RayWhisper2 follows **Clean Architecture** principles:

### Layer Structure

```
Domain Layer (Core)
    ↑
Application Layer (Use Cases)
    ↑
Infrastructure Layer (Adapters)
    ↑
Presentation Layer (UI/CLI)
```

### Key Principles

1. **Dependency Rule**: Dependencies point inward (toward domain)
2. **Single Responsibility**: Each class has one reason to change
3. **Interface Segregation**: Use specific interfaces, not general ones
4. **Dependency Inversion**: Depend on abstractions, not concretions

### Directory Structure

- `domain/`: Core business logic (no external dependencies)
  - `entities/`: Business objects with identity
  - `value_objects/`: Immutable objects without identity
  - `interfaces/`: Port interfaces (abstractions)

- `application/`: Use cases and application services
  - `use_cases/`: Application-specific business rules
  - `services/`: Application services

- `infrastructure/`: External implementations
  - Adapters for external libraries
  - Database implementations
  - API clients

- `presentation/`: User interface
  - CLI commands
  - Application orchestration

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

Follow these guidelines:

#### Code Style

- Use **type hints** for all function parameters and return values
- Follow **PEP 8** style guide
- Use **docstrings** for all public functions and classes
- Keep functions small and focused

Example:

```python
def transcribe(self, audio: AudioData, context: str | None = None) -> Transcription:
    """Transcribe audio to text.

    Args:
        audio: The audio data to transcribe.
        context: Optional context to guide transcription.

    Returns:
        Transcription: The transcription result.
    """
    # Implementation here
```

#### Testing

- Write **unit tests** for all new functionality
- Aim for **>80% code coverage**
- Use **pytest** for testing
- Mock external dependencies

Example:

```python
def test_transcription_creation() -> None:
    """Test creating a valid transcription."""
    trans = Transcription(
        text="Hello world",
        language="en",
        confidence=0.95,
        timestamp=datetime.now(),
    )
    assert trans.text == "Hello world"
```

#### Logging

- Use **loguru** for logging
- Log at appropriate levels:
  - `DEBUG`: Detailed diagnostic information
  - `INFO`: General informational messages
  - `WARNING`: Warning messages
  - `ERROR`: Error messages

Example:

```python
logger.info("Starting transcription")
logger.debug(f"Audio duration: {audio.duration:.2f}s")
logger.error(f"Failed to load model: {e}")
```

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=raywhisper --cov-report=html

# Run specific tests
pytest tests/unit/domain/
```

### 4. Check Code Quality

```bash
# Linting
ruff check src/

# Type checking
mypy src/

# Format code
ruff format src/
```

### 5. Commit Your Changes

Use conventional commit messages:

```bash
git commit -m "feat: add support for multilingual embeddings"
git commit -m "fix: resolve audio buffer overflow issue"
git commit -m "docs: update installation instructions"
git commit -m "test: add tests for reranker"
```

Commit types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test changes
- `refactor`: Code refactoring
- `style`: Code style changes
- `chore`: Build/tooling changes

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Pull Request Guidelines

### PR Title

Use conventional commit format:

```
feat: add support for custom embedding models
fix: resolve keyboard hotkey conflict on macOS
```

### PR Description

Include:

1. **What**: What does this PR do?
2. **Why**: Why is this change needed?
3. **How**: How does it work?
4. **Testing**: How was it tested?

Example:

```markdown
## What
Adds support for custom embedding models via configuration.

## Why
Users want to use different embedding models for specific use cases.

## How
- Added `custom_embedding_model` setting
- Updated ChromaVectorStore to accept custom models
- Added validation for model names

## Testing
- Added unit tests for custom model loading
- Tested with BAAI/bge-m3 and all-MiniLM-L6-v2
- Verified backward compatibility with default model
```

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Docstrings added
- [ ] No breaking changes (or documented)
- [ ] Commit messages follow convention

## Adding New Features

### 1. Define Interface (Domain Layer)

```python
# src/raywhisper/domain/interfaces/my_interface.py
from abc import ABC, abstractmethod

class IMyInterface(ABC):
    @abstractmethod
    def my_method(self, param: str) -> str:
        """Method description."""
        pass
```

### 2. Implement Adapter (Infrastructure Layer)

```python
# src/raywhisper/infrastructure/my_module/my_implementation.py
from ...domain.interfaces.my_interface import IMyInterface

class MyImplementation(IMyInterface):
    def my_method(self, param: str) -> str:
        """Implementation."""
        return f"Processed: {param}"
```

### 3. Create Use Case (Application Layer)

```python
# src/raywhisper/application/use_cases/my_use_case.py
from ...domain.interfaces.my_interface import IMyInterface

class MyUseCase:
    def __init__(self, my_interface: IMyInterface) -> None:
        self._my_interface = my_interface

    def execute(self, input: str) -> str:
        return self._my_interface.my_method(input)
```

### 4. Write Tests

```python
# tests/unit/application/test_my_use_case.py
from unittest.mock import MagicMock
from raywhisper.application.use_cases.my_use_case import MyUseCase

def test_my_use_case() -> None:
    mock_interface = MagicMock()
    mock_interface.my_method.return_value = "Result"

    use_case = MyUseCase(mock_interface)
    result = use_case.execute("input")

    assert result == "Result"
    mock_interface.my_method.assert_called_once_with("input")
```

## Reporting Bugs

### Before Reporting

1. Check existing issues
2. Try the latest version
3. Verify it's reproducible

### Bug Report Template

```markdown
**Description**
Clear description of the bug.

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happens.

**Environment**
- OS: Windows 11 / macOS Sonoma
- Python: 3.11.5
- RayWhisper: 0.1.0

**Logs**
```
Paste relevant logs here
```
```

## Feature Requests

### Feature Request Template

```markdown
**Problem**
What problem does this solve?

**Proposed Solution**
How should it work?

**Alternatives**
Other solutions considered.

**Additional Context**
Any other information.
```

## Questions?

- Open a GitHub Discussion
- Review existing code for examples

## License

By contributing, you agree that your contributions will be licensed under the MIT License.


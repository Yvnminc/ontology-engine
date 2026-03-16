"""Custom exceptions for Ontology Engine."""


class OntologyError(Exception):
    """Base exception."""


class ExtractionError(OntologyError):
    """LLM extraction failed."""


class ValidationError(OntologyError):
    """Validation check failed."""


class StorageError(OntologyError):
    """Database operation failed."""


class ConfigError(OntologyError):
    """Invalid configuration."""


class LLMError(OntologyError):
    """LLM provider returned an error."""

    def __init__(self, message: str, provider: str = "", model: str = ""):
        self.provider = provider
        self.model = model
        super().__init__(f"[{provider}/{model}] {message}")

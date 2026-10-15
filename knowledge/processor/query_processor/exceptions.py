"""Query processor exceptions."""


class QueryProcessError(Exception):
    """Base exception for query workflow failures."""

    def __init__(self, message: str, node_name: str = "", cause: Exception = None):
        self.node_name = node_name
        self.cause = cause
        super().__init__(message)

    def __str__(self):
        parts = []
        if self.node_name:
            parts.append(f"[{self.node_name}]")
        parts.append(super().__str__())
        if self.cause:
            parts.append(f"(cause: {self.cause})")
        return " ".join(parts)


class StateFieldError(QueryProcessError):
    """Raised when required state fields are missing or invalid."""

    def __init__(
        self,
        node_name: str = "",
        field_name: str = "",
        expected_type: type = None,
        message: str = "",
        cause: Exception = None,
    ):
        self.field_name = field_name
        self.expected_type = expected_type
        if not message:
            message = f"state field '{field_name}' is missing or invalid"
            if expected_type:
                message += f", expected type: {expected_type.__name__}"
        super().__init__(message, node_name=node_name, cause=cause)


class ConfigurationError(QueryProcessError):
    """Raised for invalid or missing configuration."""


class SearchError(QueryProcessError):
    """Raised when vector or hybrid search fails."""


class EmbeddingError(QueryProcessError):
    """Raised when embedding generation fails."""


class LLMError(QueryProcessError):
    """Raised when LLM calls or parsing fail."""


class StorageError(QueryProcessError):
    """Raised when storage operations fail."""


class MilvusError(StorageError):
    """Raised when Milvus operations fail."""


class ValidationError(QueryProcessError):
    """Raised when input data is invalid."""


class ItemNameConfirmError(QueryProcessError):
    """Raised when item name confirmation fails."""

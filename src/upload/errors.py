class UploadError(Exception):
    """A safe, user-displayable upload or ingestion error."""


class UnsupportedFormatError(UploadError):
    pass


class FileValidationError(UploadError):
    pass


class ParsingError(UploadError):
    pass


class NormalizationError(UploadError):
    pass

import os
from typing import Tuple
from fastapi import UploadFile


class FileValidationError(Exception):
    """Exception raised for file validation errors."""
    pass


async def validate_file(file: UploadFile, allowed_extensions: list[str], max_size: int) -> None:
    """
    Validate uploaded file for type and size.
    
    Args:
        file: The uploaded file object
        allowed_extensions: List of allowed file extensions (lowercase)
        max_size: Maximum allowed file size in bytes
        
    Raises:
        FileValidationError: If file validation fails
    """
    if not file.filename:
        raise FileValidationError("No filename provided")
    
    file_extension = os.path.splitext(file.filename)[1].lstrip(".").lower()
    
    if file_extension not in allowed_extensions:
        allowed_str = ", ".join(allowed_extensions).upper()
        raise FileValidationError(f"Invalid file type. Allowed types: {allowed_str}")
    
    if file.content_type and not file.content_type.startswith("image/"):
        raise FileValidationError("File must be an image")
    
    if file.size and file.size > max_size:
        max_size_mb = max_size / (1024 * 1024)
        raise FileValidationError(f"File size exceeds maximum limit of {max_size_mb:.1f}MB")


async def get_file_bytes(file: UploadFile) -> bytes:
    """
    Read uploaded file contents into bytes.
    
    Args:
        file: The uploaded file object
        
    Returns:
        File contents as bytes
    """
    contents = await file.read()
    await file.seek(0)
    return contents

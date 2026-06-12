import os
from typing import Tuple
from fastapi import UploadFile


class FileValidationError(Exception):
    """File validation fail ஆனா இந்த custom exception raise பண்ணு."""
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
    # Filename இல்லன்னா reject
    if not file.filename:
        raise FileValidationError("No filename provided")
    
    # "photo.JPG" → "jpg" — extension lowercase-ஆ எடு
    file_extension = os.path.splitext(file.filename)[1].lstrip(".").lower()
    
    # Extension allowed list-ல இல்லன்னா reject (e.g., .exe, .pdf)
    if file_extension not in allowed_extensions:
        allowed_str = ", ".join(allowed_extensions).upper()
        raise FileValidationError(f"Invalid file type. Allowed types: {allowed_str}")
    
    # Content-type check — "image/" இல்லாத files reject (e.g., text/plain)
    if file.content_type and not file.content_type.startswith("image/"):
        raise FileValidationError("File must be an image")
    
    # Size check — 12MB limit exceed பண்ணா reject
    if file.size and file.size > max_size:
        max_size_mb = max_size / (1024 * 1024)  # bytes → MB convert
        raise FileValidationError(f"File size exceeds maximum limit of {max_size_mb:.1f}MB")


async def get_file_bytes(file: UploadFile) -> bytes:
    """
    Read uploaded file contents into bytes.
    
    Args:
        file: The uploaded file object
        
    Returns:
        File contents as bytes
    """
    contents = await file.read()    # File-ஐ memory-ல read பண்ணு
    await file.seek(0)              # Cursor-ஐ beginning-க்கு reset பண்ணு (reusable)
    return contents

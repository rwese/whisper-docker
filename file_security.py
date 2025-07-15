"""
Security-hardened file validation for audio uploads
Protects against malicious files, format confusion, and various attack vectors
"""

import hashlib
import io
import logging
import mimetypes
import os
import re
import struct
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Magic number signatures for supported audio/video formats
AUDIO_MAGIC_NUMBERS = {
    # Audio formats
    b'\xFF\xFB': 'mp3',  # MP3 (MPEG-1 Layer 3)
    b'\xFF\xF3': 'mp3',  # MP3 (MPEG-1 Layer 3, alternative)
    b'\xFF\xF2': 'mp3',  # MP3 (MPEG-1 Layer 3, alternative)
    b'RIFF': 'wav',      # WAV (check for WAVE subtype)
    b'ftyp': 'mp4',      # MP4/M4A (at offset 4)
    b'OggS': 'ogg',      # OGG
    b'fLaC': 'flac',     # FLAC
    b'\x30\x26\xB2\x75\x8E\x66\xCF\x11': 'wma',  # WMA/ASF
    b'ID3': 'mp3',       # MP3 with ID3 tag
}

# Video container magic numbers that may contain audio
VIDEO_MAGIC_NUMBERS = {
    b'\x00\x00\x00\x14ftypqt': 'mov',   # QuickTime MOV
    b'\x00\x00\x00\x18ftypmp4': 'mp4',  # MP4
    b'\x1A\x45\xDF\xA3': 'mkv',         # Matroska/WebM
    b'RIFF': 'avi',                     # AVI (check for AVI subtype)
}

# Dangerous file signatures to reject
DANGEROUS_SIGNATURES = {
    b'MZ': 'exe',                        # Windows executable
    b'\x7FELF': 'elf',                   # Linux executable
    b'\xFE\xED\xFA': 'macho',            # macOS executable
    b'\xCF\xFA\xED\xFE': 'macho',        # macOS executable (32-bit)
    b'PK\x03\x04': 'zip',                # ZIP archive (could contain malware)
    b'PK\x05\x06': 'zip',                # ZIP archive (empty)
    b'Rar!': 'rar',                     # RAR archive
    b'\x1F\x8B': 'gzip',                # GZIP
    b'BZh': 'bzip2',                     # BZIP2
    b'\x37\x7A\xBC\xAF\x27\x1C': '7z',  # 7-Zip
    b'\xD0\xCF\x11\xE0': 'ole',         # Microsoft Office/OLE
    b'%PDF': 'pdf',                     # PDF
    b'\x89PNG': 'png',                  # PNG image
    b'\xFF\xD8\xFF': 'jpeg',            # JPEG image
    b'GIF8': 'gif',                     # GIF image
    b'<!DOCTYPE': 'html',               # HTML
    b'<html': 'html',                   # HTML
    b'<?xml': 'xml',                    # XML
    b'#!/': 'script',                   # Shell script
    b'<?php': 'php',                    # PHP script
}

# Maximum file sizes by format (in bytes)
MAX_FILE_SIZES = {
    'mp3': 100 * 1024 * 1024,    # 100MB
    'wav': 500 * 1024 * 1024,    # 500MB (uncompressed)
    'm4a': 100 * 1024 * 1024,    # 100MB
    'flac': 300 * 1024 * 1024,   # 300MB
    'ogg': 100 * 1024 * 1024,    # 100MB
    'wma': 100 * 1024 * 1024,    # 100MB
    'aac': 100 * 1024 * 1024,    # 100MB
    'mp4': 200 * 1024 * 1024,    # 200MB (video with audio)
    'mov': 200 * 1024 * 1024,    # 200MB
    'avi': 200 * 1024 * 1024,    # 200MB
    'mkv': 200 * 1024 * 1024,    # 200MB
    'webm': 100 * 1024 * 1024,   # 100MB
}

# Suspicious filename patterns
SUSPICIOUS_PATTERNS = [
    r'\.php$',
    r'\.exe$',
    r'\.scr$',
    r'\.bat$',
    r'\.cmd$',
    r'\.com$',
    r'\.pif$',
    r'\.vbs$',
    r'\.js$',
    r'\.jar$',
    r'\.sh$',
    r'\.py$',
    r'\.pl$',
    r'\.rb$',
    r'\.asp$',
    r'\.jsp$',
    r'\.cgi$',
    r'\.dll$',
    r'\.so$',
    r'\.dylib$',
    r'\\',          # Backslashes (Windows path separators)
    r'\.\.',        # Path traversal
    r'[<>:"|?*]',   # Windows forbidden chars
    r'[\x00-\x1f]', # Control characters
    r'^\.',         # Hidden files (starting with dot)
    r'~$',          # Temporary files
    r'\.exe\.',     # Double extension attacks (file.exe.mp3)
    r'\.scr\.',     # Double extension attacks
    r'\.bat\.',     # Double extension attacks
    r'\.com\.',     # Double extension attacks
    r'\.php\.',     # Double extension attacks (script.php.mp3)
    r'\.asp\.',     # Double extension attacks
    r'\.jsp\.',     # Double extension attacks
    r'\s+\.',       # Spaces before extension
    r'\s+$',        # Trailing spaces
]


class FileValidationError(Exception):
    """Custom exception for file validation failures"""
    def __init__(self, message: str, error_code: str = "INVALID_FILE"):
        super().__init__(message)
        self.error_code = error_code


class SecurityFileValidator:
    """Security-hardened file validator for audio uploads"""

    def __init__(self, max_total_size: int = 25 * 1024 * 1024):
        """Initialize validator with maximum file size"""
        self.max_total_size = max_total_size

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal and other attacks"""
        if not filename:
            raise FileValidationError("Empty filename", "EMPTY_FILENAME")

        # Store original for logging
        original_filename = filename

        # First check the original filename for path traversal patterns
        # This must be done BEFORE any sanitization
        path_traversal_patterns = [
            r'\.\.',           # Directory traversal
            r'/\.\./',         # Unix path traversal
            r'\\\.\.\\',       # Windows path traversal
            r'\.\./',          # Relative path traversal
            r'\.\.\.',         # Multiple dots
            r'/',              # Absolute paths (Unix)
            r'\\',             # Backslashes (Windows paths)
            r'^[A-Za-z]:\\',   # Windows drive letters (C:\, D:\, etc.)
            r'%2e%2e',         # URL encoded ..
            r'%2f',            # URL encoded /
            r'%5c',            # URL encoded \
            r'%00',            # URL encoded null byte
            r'\x00',           # Null byte
        ]

        for pattern in path_traversal_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                raise FileValidationError(
                    f"Path traversal pattern detected in filename: {original_filename}",
                    "PATH_TRAVERSAL"
                )

        # URL decode the filename to catch encoded attacks
        try:
            import urllib.parse
            decoded_filename = urllib.parse.unquote(filename)
            # Check decoded version for path traversal too
            for pattern in path_traversal_patterns:
                if re.search(pattern, decoded_filename, re.IGNORECASE):
                    raise FileValidationError(
                        f"Path traversal pattern detected in decoded filename: {decoded_filename}",
                        "ENCODED_PATH_TRAVERSAL"
                    )
            filename = decoded_filename
        except Exception:
            pass  # If URL decoding fails, continue with original

        # Remove path components (this should now be safe)
        filename = os.path.basename(filename)

        # Additional filename sanitization
        filename = filename.strip()  # Remove leading/trailing whitespace

        # Check for suspicious patterns in the cleaned filename
        for pattern in SUSPICIOUS_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                raise FileValidationError(
                    f"Filename contains suspicious pattern: {filename}",
                    "SUSPICIOUS_FILENAME"
                )

        # Check filename length
        if len(filename) > 255:
            raise FileValidationError(
                "Filename too long (max 255 characters)",
                "FILENAME_TOO_LONG"
            )

        # Check for empty filename after sanitization
        if not filename or filename in ['.', '..']:
            raise FileValidationError(
                "Invalid filename after sanitization",
                "INVALID_SANITIZED_FILENAME"
            )

        # Check for null bytes and control characters
        if any(ord(c) < 32 for c in filename if c not in ['\t']):
            raise FileValidationError(
                "Filename contains control characters",
                "INVALID_FILENAME_CHARS"
            )

        # Check for Windows reserved names
        windows_reserved = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
                           'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2',
                           'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']
        name_without_ext = os.path.splitext(filename)[0].upper()
        if name_without_ext in windows_reserved:
            raise FileValidationError(
                f"Windows reserved filename: {filename}",
                "RESERVED_FILENAME"
            )

        # Ensure we have a valid file extension
        name, ext = os.path.splitext(filename)
        if not ext or len(ext) < 2:  # Need at least a dot and one character
            raise FileValidationError(
                "Missing or invalid file extension",
                "INVALID_EXTENSION"
            )

        # Final validation: filename should be different from original if path traversal was attempted
        if original_filename != filename and len(original_filename) > len(filename) + 10:
            # Large difference suggests path traversal was stripped
            logger.warning(f"Significant filename sanitization applied: '{original_filename}' -> '{filename}'")

        return filename

    def detect_file_type_by_magic(self, content: bytes) -> Optional[str]:
        """Detect file type using magic number signatures"""
        if len(content) < 16:
            return None

        # Check for dangerous file types first
        for signature, file_type in DANGEROUS_SIGNATURES.items():
            if content.startswith(signature):
                raise FileValidationError(
                    f"Dangerous file type detected: {file_type}",
                    "DANGEROUS_FILE_TYPE"
                )

        # Check audio magic numbers
        for signature, file_type in AUDIO_MAGIC_NUMBERS.items():
            if signature == b'ftyp':
                # Special case for MP4/M4A - check at offset 4
                if len(content) > 8 and content[4:8] == signature:
                    # Check specific subtypes
                    if len(content) > 12:
                        subtype = content[8:12]
                        if subtype in [b'M4A ', b'mp41', b'mp42', b'isom', b'dash']:
                            return 'mp4'
            elif signature == b'RIFF':
                # Special case for RIFF containers (WAV, AVI)
                if len(content) > 12:
                    if content[8:12] == b'WAVE':
                        return 'wav'
                    elif content[8:12] == b'AVI ':
                        return 'avi'
            elif content.startswith(signature):
                return file_type

        # Check video containers
        for signature, file_type in VIDEO_MAGIC_NUMBERS.items():
            if content.startswith(signature):
                return file_type

        return None

    def validate_file_structure(self, content: bytes, detected_type: str) -> bool:
        """Perform deeper structural validation of the file"""
        try:
            if detected_type == 'wav':
                return self._validate_wav_structure(content)
            elif detected_type in ['mp4', 'm4a']:
                return self._validate_mp4_structure(content)
            elif detected_type == 'mp3':
                return self._validate_mp3_structure(content)
            elif detected_type == 'ogg':
                return self._validate_ogg_structure(content)
            elif detected_type == 'flac':
                return self._validate_flac_structure(content)
            # Add more format-specific validations as needed
            return True
        except Exception as e:
            logger.warning(f"File structure validation failed: {e}")
            return False

    def _validate_wav_structure(self, content: bytes) -> bool:
        """Validate WAV file structure"""
        if len(content) < 44:  # Minimum WAV header size
            return False

        # Check RIFF header
        if content[:4] != b'RIFF':
            return False

        # Check file size in header
        file_size = struct.unpack('<L', content[4:8])[0]
        if file_size + 8 != len(content):
            # Allow some tolerance for metadata but not too much
            if abs((file_size + 8) - len(content)) > 1024:
                return False

        # Check WAVE format
        if content[8:12] != b'WAVE':
            return False

        # Look for required chunks
        offset = 12
        found_fmt = False
        found_data = False

        while offset < len(content) - 8:
            chunk_id = content[offset:offset+4]
            chunk_size = struct.unpack('<L', content[offset+4:offset+8])[0]

            if chunk_id == b'fmt ':
                found_fmt = True
                # Validate fmt chunk
                if chunk_size < 16:  # PCM format requires at least 16 bytes
                    return False
                if offset + 8 + chunk_size > len(content):
                    return False

                # Check audio format (first 2 bytes of fmt data)
                if offset + 10 <= len(content):
                    audio_format = struct.unpack('<H', content[offset+8:offset+10])[0]
                    # Common formats: 1=PCM, 3=IEEE float, 6=A-law, 7=μ-law
                    if audio_format not in [1, 3, 6, 7, 0xFFFE]:  # 0xFFFE = EXTENSIBLE
                        return False

            elif chunk_id == b'data':
                found_data = True
                # Data chunk should have reasonable size
                if chunk_size == 0 or chunk_size > file_size:
                    return False

            # Move to next chunk (with padding alignment)
            offset += 8 + chunk_size
            if chunk_size % 2:  # WAV chunks are word-aligned
                offset += 1

        return found_fmt and found_data

    def _validate_mp4_structure(self, content: bytes) -> bool:
        """Validate MP4/M4A file structure"""
        if len(content) < 32:
            return False

        # Check ftyp box
        box_size = struct.unpack('>L', content[:4])[0]
        if box_size < 8 or box_size > len(content):
            return False

        box_type = content[4:8]
        if box_type != b'ftyp':
            return False

        return True

    def _validate_mp3_structure(self, content: bytes) -> bool:
        """Validate MP3 file structure"""
        if len(content) < 10:  # Need at least 10 bytes for meaningful validation
            return False

        # Skip ID3 tag if present
        offset = 0
        if content.startswith(b'ID3'):
            if len(content) < 10:
                return False
            # Get ID3 tag size (synchsafe integer)
            tag_size_bytes = content[6:10]
            tag_size = ((tag_size_bytes[0] & 0x7F) << 21) | \
                      ((tag_size_bytes[1] & 0x7F) << 14) | \
                      ((tag_size_bytes[2] & 0x7F) << 7) | \
                      (tag_size_bytes[3] & 0x7F)
            offset = 10 + tag_size

        if offset >= len(content):
            return False

        # Check for MP3 frame header and validate it
        if offset + 4 > len(content):
            return False

        frame_header = content[offset:offset+4]

        # Check sync word (11 bits set)
        if (frame_header[0] != 0xFF) or ((frame_header[1] & 0xE0) != 0xE0):
            return False

        # Check MPEG version and layer (should not be reserved values)
        version = (frame_header[1] >> 3) & 0x03
        layer = (frame_header[1] >> 1) & 0x03

        if version == 1 or layer == 0:  # Reserved values
            return False

        # Check bitrate (should not be free or bad value)
        bitrate_index = (frame_header[2] >> 4) & 0x0F
        if bitrate_index == 0 or bitrate_index == 15:  # Free or bad bitrate
            return False

        # Check sampling rate (should not be reserved)
        sampling_rate = (frame_header[2] >> 2) & 0x03
        if sampling_rate == 3:  # Reserved value
            return False

        # Additional validation: file should be mostly MP3 frames
        # Check if we have reasonable frame distribution
        frame_count = 0
        check_offset = offset
        while check_offset < len(content) - 4 and frame_count < 3:
            if content[check_offset] == 0xFF and (content[check_offset + 1] & 0xE0) == 0xE0:
                frame_count += 1
                # Simple frame length estimation (this is approximate)
                check_offset += 144  # Rough average frame size
            else:
                check_offset += 1

        return frame_count >= 1  # At least one valid frame

    def _validate_ogg_structure(self, content: bytes) -> bool:
        """Validate OGG file structure"""
        if len(content) < 27:  # Minimum OGG page header size
            return False

        # Check OGG signature
        if content[:4] != b'OggS':
            return False

        # Check version
        if content[4] != 0:
            return False

        return True

    def _validate_flac_structure(self, content: bytes) -> bool:
        """Validate FLAC file structure"""
        if len(content) < 4:
            return False

        # Check FLAC signature
        if content[:4] != b'fLaC':
            return False

        return True

    def calculate_entropy(self, data: bytes, sample_size: int = 1024) -> float:
        """Calculate entropy to detect encrypted/compressed malicious content"""
        if len(data) < sample_size:
            sample_size = len(data)

        sample = data[:sample_size]

        # Count byte frequencies
        frequencies = {}
        for byte in sample:
            frequencies[byte] = frequencies.get(byte, 0) + 1

        # Calculate entropy using Shannon entropy formula
        import math
        entropy = 0.0
        for count in frequencies.values():
            probability = count / len(sample)
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def check_file_anomalies(self, content: bytes, filename: str) -> List[str]:
        """Check for various file anomalies that might indicate malicious content"""
        anomalies = []

        # Check entropy (high entropy might indicate encryption/compression)
        entropy = self.calculate_entropy(content)
        if entropy > 7.5:  # Very high entropy
            anomalies.append(f"High entropy detected: {entropy:.2f}")

        # Check for embedded executables
        if b'This program cannot be run in DOS mode' in content:
            anomalies.append("Embedded Windows executable detected")

        if b'\x7FELF' in content[100:]:  # ELF signature not at start
            anomalies.append("Embedded Linux executable detected")

        # Check for script injection
        script_patterns = [
            b'<script',
            b'javascript:',
            b'eval(',
            b'system(',
            b'exec(',
            b'shell_exec(',
            b'passthru(',
            b'<?php',
            b'#!/bin/',
        ]

        for pattern in script_patterns:
            if pattern in content.lower():
                anomalies.append(f"Script injection pattern detected: {pattern.decode('utf-8', errors='ignore')}")

        # Check for unusual file size vs extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in ['.wav'] and len(content) < 1000:
            anomalies.append("Suspiciously small WAV file")

        # Check for polyglot files (files that are valid in multiple formats)
        if content.startswith(b'PK') and b'ftyp' in content[:100]:
            anomalies.append("Possible polyglot file (ZIP + MP4)")

        return anomalies

    def validate_file(self, content: bytes, filename: str) -> Dict[str, any]:
        """
        Comprehensive file validation with security hardening

        Returns:
            dict: Validation result with metadata

        Raises:
            FileValidationError: If validation fails
        """
        # Sanitize filename first
        clean_filename = self.sanitize_filename(filename)

        # Check file size
        if len(content) == 0:
            raise FileValidationError("Empty file", "EMPTY_FILE")

        if len(content) > self.max_total_size:
            raise FileValidationError(
                f"File too large: {len(content)} bytes (max: {self.max_total_size})",
                "FILE_TOO_LARGE"
            )

        # Detect file type by magic number
        detected_type = self.detect_file_type_by_magic(content)
        if not detected_type:
            raise FileValidationError(
                "Unrecognized or unsupported file format",
                "UNSUPPORTED_FORMAT"
            )

        # Check against format-specific size limits
        if detected_type in MAX_FILE_SIZES:
            max_size = MAX_FILE_SIZES[detected_type]
            if len(content) > max_size:
                raise FileValidationError(
                    f"File too large for {detected_type}: {len(content)} bytes (max: {max_size})",
                    "FORMAT_SIZE_EXCEEDED"
                )

        # Validate file extension matches detected type
        file_ext = os.path.splitext(clean_filename)[1].lower().lstrip('.')
        expected_extensions = {
            'mp3': ['mp3'],
            'wav': ['wav'],
            'mp4': ['mp4', 'm4a', 'mp4a'],
            'ogg': ['ogg', 'oga'],
            'flac': ['flac'],
            'wma': ['wma'],
            'avi': ['avi'],
            'mov': ['mov'],
            'mkv': ['mkv'],
            'webm': ['webm'],
        }

        if detected_type in expected_extensions:
            if file_ext not in expected_extensions[detected_type]:
                logger.warning(f"Extension mismatch: {file_ext} vs detected {detected_type}")
                # Don't reject but log for monitoring

        # Perform structural validation
        if not self.validate_file_structure(content, detected_type):
            raise FileValidationError(
                f"Invalid {detected_type} file structure",
                "INVALID_STRUCTURE"
            )

        # Check for anomalies
        anomalies = self.check_file_anomalies(content, clean_filename)
        if anomalies:
            # Log anomalies but don't necessarily reject
            logger.warning(f"File anomalies detected in {clean_filename}: {anomalies}")

            # Reject if critical anomalies found
            critical_keywords = ['executable', 'script injection', 'polyglot']
            for anomaly in anomalies:
                if any(keyword in anomaly.lower() for keyword in critical_keywords):
                    raise FileValidationError(
                        f"Critical security anomaly: {anomaly}",
                        "SECURITY_VIOLATION"
                    )

        # Calculate file hash for deduplication/tracking
        file_hash = hashlib.sha256(content).hexdigest()

        return {
            'valid': True,
            'filename': clean_filename,
            'detected_type': detected_type,
            'file_size': len(content),
            'file_hash': file_hash,
            'anomalies': anomalies,
        }


# Singleton validator instance
validator = SecurityFileValidator()


def validate_uploaded_file(content: bytes, filename: str) -> Dict[str, any]:
    """
    Main entry point for file validation

    Args:
        content: File content as bytes
        filename: Original filename

    Returns:
        dict: Validation result

    Raises:
        FileValidationError: If validation fails
    """
    return validator.validate_file(content, filename)


def get_safe_filename(filename: str) -> str:
    """Get sanitized filename"""
    return validator.sanitize_filename(filename)

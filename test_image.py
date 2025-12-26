#!/usr/bin/env python3
"""Test image extraction functionality"""

from kiro_gateway.converters import _extract_images_from_content
import json

# Test Anthropic format
anthropic_content = [
    {
        "type": "text",
        "text": "What color is this?"
    },
    {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        }
    }
]

print("Testing Anthropic format image extraction...")
images = _extract_images_from_content(anthropic_content)
print(f"Extracted {len(images)} image(s)")
print(json.dumps(images, indent=2))

# Test OpenAI format
openai_content = [
    {
        "type": "text",
        "text": "What color is this?"
    },
    {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        }
    }
]

print("\nTesting OpenAI format image extraction...")
images = _extract_images_from_content(openai_content)
print(f"Extracted {len(images)} image(s)")
print(json.dumps(images, indent=2))

"""
Text preprocessing utilities for handling Unicode and special characters
"""
import re
import json
import html


def decode_unicode_escapes(text):
    """
    Decode Unicode escape sequences like \u00a1, \ud83d\udc4b
    
    Examples:
        \u00a1Hola! → ¡Hola!
        \ud83d\udc4b → 👋
        \u00f1 → ñ
    """
    if not isinstance(text, str):
        return text
    
    try:
        if '\\u' in text:
            try:
                decoded = json.loads(f'"{text}"')
                return decoded
            except:
                pass
        
        try:
            decoded = text.encode().decode('unicode-escape')
            return decoded
        except:
            pass
        
        return text
    except Exception as e:
        print(f"Warning: Could not decode unicode in text: {str(e)[:50]}")
        return text


def decode_html_entities(text):
    """
    Decode HTML entities like &amp;, &lt;, &gt;, &#39;
    
    Examples:
        &amp; → &
        &lt; → <
        &gt; → >
        &#39; → '
    """
    if not isinstance(text, str):
        return text
    
    try:
        return html.unescape(text)
    except:
        return text


def normalize_whitespace(text):
    """
    Normalize whitespace (multiple spaces, tabs, newlines)
    """
    if not isinstance(text, str):
        return text
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def clean_text(text, decode_unicode=True, decode_html=True, normalize_ws=False):
    """
    Clean text by decoding Unicode escapes and HTML entities
    
    Args:
        text: Input text
        decode_unicode: Decode Unicode escape sequences
        decode_html: Decode HTML entities
        normalize_ws: Normalize whitespace
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return text
    
    if decode_unicode:
        text = decode_unicode_escapes(text)
    
    if decode_html:
        text = decode_html_entities(text)
    
    if normalize_ws:
        text = normalize_whitespace(text)
    
    return text


def preprocess_example(example, clean_input=True, clean_target=False):
    """
    Preprocess a training/test example
    
    Args:
        example: Dict with 'input' and optionally 'target'
        clean_input: Clean the input text
        clean_target: Clean the target JSON (usually not needed)
    
    Returns:
        Preprocessed example
    """
    processed = example.copy()
    
    if clean_input and 'input' in processed:
        processed['input'] = clean_text(
            processed['input'],
            decode_unicode=True,
            decode_html=True,
            normalize_ws=False  
        )
    
    if clean_target and 'target' in processed:
        if isinstance(processed['target'], str):
            processed['target'] = clean_text(
                processed['target'],
                decode_unicode=True,
                decode_html=True,
                normalize_ws=False
            )
    
    return processed


if __name__ == "__main__":
    test_cases = [
        r"\u00a1Hola!\ud83d\udc4b",  
        r"Espa\u00f1a",  
        r"Caf\u00e9",  
        r"\u00bfC\u00f3mo est\u00e1s?",  
        "&amp; &lt; &gt; &#39;",  
    ]
    
    print("Testing text preprocessing:")
    print("="*60)
    
    for test in test_cases:
        cleaned = clean_text(test)
        print(f"Original: {test}")
        print(f"Cleaned:  {cleaned}")
        print()

import json
import random
import re
from typing import Dict, List, Any
import copy

class DataAugmenter:
    """
    Data augmentation strategies for text-to-JSON canonicalization tasks.
    """
    
    def __init__(self, augmentation_factor=2):
        """
        Args:
            augmentation_factor: How many augmented versions to create per original example
        """
        self.augmentation_factor = augmentation_factor
        
        # Synonyms for common purchase order terms (Spanish)
        self.synonyms = {
            "comprar": ["adquirir", "pedir", "solicitar", "ordenar"],
            "necesito": ["requiero", "quiero", "deseo", "me hace falta"],
            "enviar": ["mandar", "despachar", "remitir", "entregar"],
            "urgente": ["prioritario", "inmediato", "rápido", "express"],
            "unidades": ["piezas", "items", "productos", "artículos"],
            "precio": ["costo", "valor", "monto", "importe"],
            "total": ["suma", "monto total", "importe total"],
        }
        
        # Paraphrasing templates
        self.paraphrase_templates = [
            lambda text: text,  # Original
            lambda text: text.replace(".", ",") if "." in text else text,
            lambda text: text.replace(",", " y") if "," in text else text,
        ]
    
    def augment_dataset(self, dataset):
        """
        Augment an entire dataset.
        
        Args:
            dataset: HuggingFace Dataset with 'input' and 'target' columns
            
        Returns:
            Augmented dataset
        """
        augmented_data = []
        
        for example in dataset:
            # Add original
            augmented_data.append(example)
            
            # Create augmented versions
            for _ in range(self.augmentation_factor - 1):
                aug_example = self.augment_example(example)
                augmented_data.append(aug_example)
        
        # Convert back to dataset format
        from datasets import Dataset
        import pandas as pd
        df = pd.DataFrame(augmented_data)
        return Dataset.from_pandas(df)
    
    def augment_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply random augmentation to a single example.
        
        Args:
            example: Dict with 'input' (text) and 'target' (JSON string)
            
        Returns:
            Augmented example
        """
        augmented = copy.deepcopy(example)
        
        # Choose random augmentation strategy
        strategy = random.choice([
            'synonym_replacement',
            'word_order_variation',
            'punctuation_variation',
            'number_format_variation',
            'case_variation',
            'whitespace_variation',
        ])
        
        if strategy == 'synonym_replacement':
            augmented['input'] = self._synonym_replacement(augmented['input'])
        elif strategy == 'word_order_variation':
            augmented['input'] = self._word_order_variation(augmented['input'])
        elif strategy == 'punctuation_variation':
            augmented['input'] = self._punctuation_variation(augmented['input'])
        elif strategy == 'number_format_variation':
            augmented['input'] = self._number_format_variation(augmented['input'])
        elif strategy == 'case_variation':
            augmented['input'] = self._case_variation(augmented['input'])
        elif strategy == 'whitespace_variation':
            augmented['input'] = self._whitespace_variation(augmented['input'])
        
        return augmented
    
    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms"""
        words = text.split()
        new_words = []
        
        for word in words:
            word_lower = word.lower()
            replaced = False
            
            for key, synonyms in self.synonyms.items():
                if key in word_lower:
                    if random.random() < 0.3:  # 30% chance to replace
                        synonym = random.choice(synonyms)
                        new_word = word_lower.replace(key, synonym)
                        new_words.append(new_word)
                        replaced = True
                        break
            
            if not replaced:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def _word_order_variation(self, text: str) -> str:
        """Slightly vary word order while maintaining meaning"""
        # Split by common separators
        parts = re.split(r'([,;.])', text)
        
        # Shuffle non-punctuation parts with low probability
        if len(parts) > 3 and random.random() < 0.3:
            non_punct = [p for p in parts if p not in [',', ';', '.']]
            if len(non_punct) > 1:
                # Swap two random parts
                idx1, idx2 = random.sample(range(len(non_punct)), 2)
                non_punct[idx1], non_punct[idx2] = non_punct[idx2], non_punct[idx1]
                
                # Reconstruct
                result = []
                non_punct_idx = 0
                for p in parts:
                    if p in [',', ';', '.']:
                        result.append(p)
                    else:
                        result.append(non_punct[non_punct_idx])
                        non_punct_idx += 1
                return ''.join(result)
        
        return text
    
    def _punctuation_variation(self, text: str) -> str:
        """Vary punctuation"""
        variations = [
            (r'\s*,\s*', ', '),  # Normalize comma spacing
            (r'\s*\.\s*', '. '),  # Normalize period spacing
            (r'\s+', ' '),  # Normalize whitespace
        ]
        
        result = text
        for pattern, replacement in variations:
            if random.random() < 0.3:
                result = re.sub(pattern, replacement, result)
        
        return result.strip()
    
    def _number_format_variation(self, text: str) -> str:
        """Vary number formats (e.g., 1000 vs 1,000 vs mil)"""
        # Find numbers
        def replace_number(match):
            num = match.group()
            if random.random() < 0.3:
                # Add/remove thousand separators
                try:
                    value = int(num.replace(',', ''))
                    if random.random() < 0.5:
                        return f"{value:,}"  # With comma
                    else:
                        return str(value)  # Without comma
                except:
                    return num
            return num
        
        return re.sub(r'\d[\d,]*', replace_number, text)
    
    def _case_variation(self, text: str) -> str:
        """Vary capitalization"""
        if random.random() < 0.3:
            # Randomly choose case variation
            choice = random.choice(['lower', 'title', 'original'])
            if choice == 'lower':
                return text.lower()
            elif choice == 'title':
                return text.title()
        return text
    
    def _whitespace_variation(self, text: str) -> str:
        """Add/remove extra whitespace"""
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Randomly add extra spaces
        if random.random() < 0.2:
            words = text.split()
            if len(words) > 2:
                idx = random.randint(0, len(words) - 1)
                words[idx] = words[idx] + '  '  # Double space
            text = ' '.join(words)
        
        return text.strip()
    
    def back_translation_augmentation(self, text: str, target_lang='en') -> str:
        """
        Back-translation augmentation (requires translation API).
        This is a placeholder - implement with actual translation service if needed.
        """
        # Would require: Spanish -> English -> Spanish
        # Using services like Google Translate API, DeepL, etc.
        # For now, return original
        return text
    
    def contextual_word_embeddings_augmentation(self, text: str) -> str:
        """
        Use contextual embeddings to replace words with similar ones.
        This is a placeholder - would require a language model.
        """
        # Would use models like BERT to find contextually similar words
        return text


def augment_json_variations(json_str: str) -> List[str]:
    """
    Create variations of JSON formatting while maintaining semantic equivalence.
    
    Args:
        json_str: JSON string
        
    Returns:
        List of JSON string variations
    """
    try:
        data = json.loads(json_str)
    except:
        return [json_str]
    
    variations = []
    
    # Original
    variations.append(json_str)
    
    # Compact format
    variations.append(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
    
    # Pretty format with different indents
    variations.append(json.dumps(data, ensure_ascii=False, indent=2))
    variations.append(json.dumps(data, ensure_ascii=False, indent=4))
    
    # Different key ordering (if dict)
    if isinstance(data, dict):
        sorted_data = {k: data[k] for k in sorted(data.keys())}
        variations.append(json.dumps(sorted_data, ensure_ascii=False))
    
    return variations


if __name__ == "__main__":
    # Test augmentation
    augmenter = DataAugmenter(augmentation_factor=3)
    
    test_example = {
        'input': 'Necesito comprar 100 unidades de producto A, precio 50 pesos cada uno, envío urgente.',
        'target': '{"producto": "A", "cantidad": 100, "precio_unitario": 50, "urgente": true}'
    }
    
    print("Original:")
    print(test_example['input'])
    print()
    
    print("Augmented versions:")
    for i in range(5):
        aug = augmenter.augment_example(test_example)
        print(f"{i+1}. {aug['input']}")

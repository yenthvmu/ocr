#!/usr/bin/env python3
"""
text_corrector.py - Simple Vietnamese Text Corrector
Uses underthesea for context-aware correction
"""

import re
import os

try:
    from underthesea import word_tokenize, pos_tag
    UNDERSEA_AVAILABLE = True
except ImportError:
    UNDERSEA_AVAILABLE = False
    print("⚠ Install underthesea: pip install underthesea")

class VietnameseCorrector:
    def __init__(self):
        # Common OCR errors in Vietnamese
        self.ocr_errors = {
            'băng': 'bằng',      # context: before "tiếng"
            'đề': 'để',          # context: preposition
            'cân': 'cần',        # context: before verb
            'nam': 'nhằm',       # context: purpose
            'thiếu số': 'thiểu số',
            'chính phù': 'chính phủ',
            'chính phú': 'chính phủ',
            'dê': 'dễ',
            'tiếp thụ': 'tiếp thu',
            'kiên thức': 'kiến thức',
            'học tạp': 'học tập',
            'phố biến': 'phổ biến',
            'giao dich': 'giao dịch',
            'quốc tê': 'quốc tế',
            'liên tụ': 'liên tục',
            'văn hoá': 'văn hóa',
            'bản sác': 'bản sắc',
        }
        
    def fix_with_dictionary(self, text: str) -> str:
        """Fix common OCR errors using dictionary"""
        for wrong, correct in self.ocr_errors.items():
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, text, flags=re.IGNORECASE)
        return text
    
    def fix_with_underthesea(self, text: str) -> str:
        """Use underthesea for context-aware correction"""
        if not UNDERSEA_AVAILABLE:
            return text
        
        try:
            # Tokenize and POS tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(text)
            
            # Context-based corrections
            corrected_tokens = []
            for i, (token, pos) in enumerate(pos_tags):
                current_token = token
                
                # Rule 1: "băng" before "tiếng" -> "bằng"
                if token.lower() == 'băng' and i+1 < len(tokens):
                    next_token = tokens[i+1].lower()
                    if 'tiếng' in next_token:
                        current_token = 'bằng' if token.islower() else 'Bằng'
                
                # Rule 2: "đề" as preposition -> "để"
                elif token.lower() == 'đề' and pos == 'P':
                    current_token = 'để'
                
                # Rule 3: "cân" before verb -> "cần"
                elif token.lower() == 'cân' and i+1 < len(pos_tags):
                    next_pos = pos_tags[i+1][1]
                    if next_pos.startswith('V'):  # Verb
                        current_token = 'cần'
                
                # Rule 4: Fix line numbers (22 -> 2.)
                elif i == 0 and token.isdigit() and len(token) > 1:
                    current_token = token[0] + '.'
                
                corrected_tokens.append(current_token)
            
            return ' '.join(corrected_tokens)
            
        except Exception as e:
            print(f"⚠ underthesea error: {e}")
            return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Fix line numbers at start of lines
        lines = []
        for line in text.split('\n'):
            line = re.sub(r'^(\d{2,})\s+', lambda m: f"{m.group(1)[0]}. ", line)
            line = re.sub(r'^(\d+)\.\s*', r'\1. ', line)
            lines.append(line.strip())
        
        # Join lines that shouldn't be broken
        text = '\n'.join(lines)
        text = re.sub(r'([a-z,])\n([a-z])', r'\1 \2', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def correct(self, text: str) -> str:
        """Main correction function"""
        if not text or not text.strip():
            return text
        
        # Step 1: Dictionary-based fixes
        text = self.fix_with_dictionary(text)
        
        # Step 2: Context-aware fixes with underthesea
        text = self.fix_with_underthesea(text)
        
        # Step 3: Clean and normalize
        text = self.clean_text(text)
        
        return text

# Global corrector instance
_corrector = VietnameseCorrector()

def correct_ocr_text(text: str) -> str:
    """Public function to correct OCR text"""
    return _corrector.correct(text)

def process_result_file(file_path: str = "result.txt") -> str:
    """Process result.txt file and save corrected version"""
    if not os.path.exists(file_path):
        return ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original = f.read()
        
        corrected = correct_ocr_text(original)
        
        # Save back if changed
        if corrected != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(corrected)
            print(f"✅ Corrected {file_path}")
        
        return corrected
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return ""

if __name__ == "__main__":
    if os.path.exists("result.txt"):
        process_result_file()
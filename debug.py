import main
import re

text = "Akaki Kality Mary"
text = re.sub(r'Y\b', 'I', text)
print(f"Transliterating: {main.transliterate_to_amharic(text)}")

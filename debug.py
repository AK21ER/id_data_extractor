import re

def test_trans(text):
    text = text.upper()
    text = text.replace("YOHANNES", "YOHHNIS")
    text = text.replace("HAILU", "XAYLU")
    text = text.replace("AKAKI", "AQAQI")
    text = re.sub(r'\bKA', 'QA', text)
    text = re.sub(r'\bWO', 'W', text)
    return text

print(test_trans("Akaki Kality"))
print(test_trans("Wondwossen Kaleab"))

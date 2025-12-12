## python3 translation_class.py
from deep_translator import GoogleTranslator, single_detection

class TranslatorPipeline:
    def __init__(self):
        pass

    def detect_language(self, text, DETECT_LANG_KEY):
        """Detect user input language."""
        lang = single_detection(text, api_key=DETECT_LANG_KEY)
        return lang

    def translate_to_english(self, text):
        """Translate any language to English."""
        return GoogleTranslator(source='auto', target='en').translate(text)

    def translate_from_english(self, text, target_lang):
        """Translate English text to user's original language."""
        return GoogleTranslator(source='en', target=target_lang).translate(text)

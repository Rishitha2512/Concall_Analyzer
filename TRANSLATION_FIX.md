# Translation Fix

The previous professional ZIP had reverted to the old `deep-translator` implementation in `research_modules.py`.
This version restores the working local Meta NLLB-200 translator.

## Install
```powershell
pip install -r requirements.txt
```

## Run
```powershell
python -m streamlit run app.py
```

On the first translation, Hugging Face downloads `facebook/nllb-200-distilled-600M`. This is expected.
After the model is downloaded, translation runs locally and does not depend on GoogleTranslator/MyMemory rate limits.

If you previously had conflicting packages in the global Python installation, use a project virtual environment.

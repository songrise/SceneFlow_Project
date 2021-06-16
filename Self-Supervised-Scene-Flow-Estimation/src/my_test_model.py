
import importlib


model = "model_concat_upsa"

MODEL = importlib.import_module(model)  # import network module

print(MODEL)
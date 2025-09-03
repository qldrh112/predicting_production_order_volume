import importlib, traceback
spec = importlib.util.find_spec('langchain_mcp')
print('spec =', spec)
try:
    mod = importlib.import_module('langchain_mcp')
    attrs = dir(mod)
    matches = [n for n in attrs if any(k in n.lower() for k in ['mcp','stdio','toolkit','server'])]
    print('matches =', matches)
except Exception as e:
    print('import_error:', e)
    traceback.print_exc()

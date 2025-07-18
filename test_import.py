# test_import.py
try:
    from pricer import OptionsPricer
    print("Import successful!")
    print(OptionsPricer)
except ImportError as e:
    print(f"Import failed: {e}")
    
# Also try:
import pricer
print("Available in pricer module:")
print([attr for attr in dir(pricer) if not attr.startswith('_')])
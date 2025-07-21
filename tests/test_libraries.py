# test_libraries.py
try:
    from py_vollib.black_scholes import black_scholes
    print("✅ py_vollib is working")
    
    # Test calculation
    price = black_scholes('c', 100, 100, 0.25, 0.05, 0.3)
    print(f"Test BS price: {price}")
    
except ImportError as e:
    print(f"❌ py_vollib not available: {e}")
    print("Install with: pip install py_vollib")
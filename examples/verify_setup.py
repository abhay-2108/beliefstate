#!/usr/bin/env python3
"""
Verify BeliefState + Ollama setup is correct.
Run: python examples/verify_setup.py
"""

import sys
import asyncio


def check_imports():
    """Check all required imports."""
    print("1️⃣  Checking imports...")
    try:
        from beliefstate import BeliefTracker, TrackerConfig
        from beliefstate.adapters.ollama import OllamaAdapter
        from openai import AsyncOpenAI
        import streamlit
        print("   ✅ All imports successful")
        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False


def check_ollama_connection():
    """Check if Ollama is running."""
    print("\n2️⃣  Checking Ollama connection...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"   ✅ Ollama running with {len(models)} models")
            
            # Check for required models
            model_names = [m.get("name", "") for m in models]
            if "qwen2.5:7b" in model_names:
                print("   ✅ qwen2.5:7b found")
            else:
                print("   ⚠️  qwen2.5:7b not found (run: ollama pull qwen2.5:7b)")
            
            if "nomic-embed-text:v1.5" in model_names or "nomic-embed-text" in model_names:
                print("   ✅ nomic-embed-text found")
            else:
                print("   ⚠️  nomic-embed-text:v1.5 not found (run: ollama pull nomic-embed-text:v1.5)")
            
            return True
        else:
            print(f"   ❌ Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Ollama not running: {e}")
        print("   → Start Ollama: ollama serve")
        return False


async def test_adapter():
    """Test OllamaAdapter initialization."""
    print("\n3️⃣  Testing OllamaAdapter...")
    try:
        from beliefstate.adapters.ollama import OllamaAdapter
        
        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            embed_model="nomic-embed-text:v1.5",
            host="http://localhost",
            port=11434,
        )
        print("   ✅ OllamaAdapter initialized")
        
        # Test health check
        is_healthy = await adapter.health_check()
        if is_healthy:
            print("   ✅ Health check passed")
            return True
        else:
            print("   ❌ Health check failed")
            return False
    except Exception as e:
        print(f"   ❌ Adapter test failed: {e}")
        return False


async def test_tracker():
    """Test BeliefTracker initialization."""
    print("\n4️⃣  Testing BeliefTracker...")
    try:
        from beliefstate import BeliefTracker, TrackerConfig
        from beliefstate.adapters.ollama import OllamaAdapter
        
        config = TrackerConfig(
            enable_background_tasks=False,
            max_beliefs=20,
        )
        
        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            embed_model="nomic-embed-text:v1.5",
            host="http://localhost",
            port=11434,
        )
        
        tracker = BeliefTracker(config=config, adapter=adapter)
        tracker.set_session("test_session")
        print("   ✅ BeliefTracker initialized")
        return True
    except Exception as e:
        print(f"   ❌ Tracker test failed: {e}")
        return False


async def test_decorator():
    """Test @tracker.wrap decorator pattern."""
    print("\n5️⃣  Testing @tracker.wrap decorator...")
    try:
        from beliefstate import BeliefTracker, TrackerConfig
        from beliefstate.adapters.ollama import OllamaAdapter
        
        config = TrackerConfig(enable_background_tasks=False)
        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            embed_model="nomic-embed-text:v1.5",
            host="http://localhost",
            port=11434,
        )
        tracker = BeliefTracker(config=config, adapter=adapter)
        tracker.set_session("test_session")
        
        # Test decorator pattern
        @tracker.wrap
        async def dummy_function():
            return "test"
        
        print("   ✅ @tracker.wrap decorator applied successfully")
        return True
    except Exception as e:
        print(f"   ❌ Decorator test failed: {e}")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("   BeliefState + Ollama Setup Verification")
    print("=" * 60)
    
    results = []
    
    # Synchronous checks
    results.append(("Imports", check_imports()))
    results.append(("Ollama Connection", check_ollama_connection()))
    
    # Async checks
    results.append(("OllamaAdapter", asyncio.run(test_adapter())))
    results.append(("BeliefTracker", asyncio.run(test_tracker())))
    results.append(("@tracker.wrap", asyncio.run(test_decorator())))
    
    # Summary
    print("\n" + "=" * 60)
    print("   SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"   {status} {name}")
    
    print(f"\n   {passed}/{total} checks passed")
    
    if passed == total:
        print("\n   ✅ Everything is ready!")
        print("\n   Run: streamlit run examples/streamlit_app_simple.py")
        return 0
    else:
        print("\n   ❌ Some checks failed")
        print("   Fix issues above and try again")
        return 1


if __name__ == "__main__":
    sys.exit(main())

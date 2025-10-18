#!/usr/bin/env python3
"""
Test script to verify pst_tools modules work correctly.
"""

import sys

def test_banner():
    """Test banner display module."""
    print("[*] Testing banner module...")
    try:
        from pst_tools import banner
        banner.display()
        print("[+] Banner module: OK\n")
        return True
    except Exception as e:
        print(f"[!] Banner module failed: {e}\n")
        return False


def test_id_generator():
    """Test ID generator module."""
    print("[*] Testing id_generator module...")
    try:
        from pst_tools import id_generator
        
        # generate a few IDs
        id1 = id_generator.generate()
        id2 = id_generator.generate()
        id3 = id_generator.generate(length=12)
        
        print(f"    Generated ID 1: {id1}")
        print(f"    Generated ID 2: {id2}")
        print(f"    Generated ID 3 (length=12): {id3}")
        
        # verify uniqueness
        if id1 != id2:
            print("[+] ID generator module: OK\n")
            return True
        else:
            print("[!] IDs are not unique\n")
            return False
            
    except Exception as e:
        print(f"[!] ID generator module failed: {e}\n")
        return False


def test_pst_parser():
    """Test PST parser module (import only, not parsing)."""
    print("[*] Testing pst_parser module...")
    try:
        from pst_tools import pst_parser
        
        # check if required functions exist
        assert hasattr(pst_parser, 'extract_emails')
        
        print("    Found: extract_emails()")
        print("[+] PST parser module: OK\n")
        return True
        
    except ImportError as e:
        print(f"[!] PST parser module failed (pypff not installed): {e}\n")
        print("    This is expected if pypff is not installed yet")
        return False
    except Exception as e:
        print(f"[!] PST parser module failed: {e}\n")
        return False


def main():
    """Run all module tests."""
    print("\n" + "="*60)
    print("PST Tools Module Tests")
    print("="*60 + "\n")
    
    results = []
    
    results.append(("Banner", test_banner()))
    results.append(("ID Generator", test_id_generator()))
    results.append(("PST Parser", test_pst_parser()))
    
    print("="*60)
    print("Test Results:")
    print("="*60)
    
    for module_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status:8} - {module_name}")
    
    print("="*60 + "\n")
    
    # check if all tests passed
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("[+] All tests passed!")
        return 0
    else:
        print("[!] Some tests failed - check dependencies")
        return 1


if __name__ == "__main__":
    sys.exit(main())


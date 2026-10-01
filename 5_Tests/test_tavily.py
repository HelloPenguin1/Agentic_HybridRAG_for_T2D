#!/usr/bin/env python3
"""
Test script for Tavily web search integration with LangChain
"""

import os
from langchain_community.tools.tavily_search import TavilySearchResults


def test_tavily_search():
    """Test Tavily search with medical domain filtering"""

    # Check if API key is set
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("❌ ERROR: TAVILY_API_KEY environment variable not set")
        print("Set it with: export TAVILY_API_KEY='your-key-here'")
        return False

    print("✓ API key found")
    print(f"  Key preview: {api_key[:8]}...")

    # Initialize the search tool
    try:
        web_search_tool = TavilySearchResults(
            k=3,
            search_depth="advanced",
            include_domains=[
                "pubmed.ncbi.nlm.nih.gov",
                "diabetes.org",
                "nih.gov",
                "mayoclinic.org",
            ],
        )
        print("✓ TavilySearchResults initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Tavily: {e}")
        return False

    # Test query related to diabetes medication
    test_query = "What is hyperglycemia"

    print(f"\n📋 Running test query: '{test_query}'")
    print("=" * 60)

    try:
        # Execute search
        results = web_search_tool.invoke({"query": test_query})

        print(f"✓ Search completed successfully")
        print(f"  Results type: {type(results)}")
        print(
            f"  Number of results: {len(results) if isinstance(results, list) else 'N/A'}"
        )

        # Display results
        print("\n📊 RESULTS:")
        print("=" * 60)

        if isinstance(results, list):
            for idx, result in enumerate(results, 1):
                print(f"\n[Result {idx}]")
                if isinstance(result, dict):
                    print(f"  URL: {result.get('url', 'N/A')}")
                    print(f"  Title: {result.get('title', 'N/A')}")
                    content = result.get("content", "N/A")
                    print(
                        f"  Content: {content[:200]}..."
                        if len(content) > 200
                        else f"  Content: {content}"
                    )
                else:
                    print(f"  {result}")
        else:
            print(results)

        print("\n" + "=" * 60)
        print("✅ TEST PASSED: Tavily search is working correctly")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: Search execution error")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("TAVILY WEB SEARCH TEST")
    print("=" * 60)
    print()

    success = test_tavily_search()

    print()
    exit(0 if success else 1)

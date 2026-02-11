# ============================================================================
# CORRECTED CYPHER CHAIN SETUP - Copy this into your notebook
# ============================================================================

# Cell 1: Create the Cypher chain (CORRECTED VERSION)
"""
# Create the Cypher chain with allow_dangerous_requests flag
cypher_chain = GraphCypherQAChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    graph=graph,
    verbose=True,
    validate_cypher=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    top_k=10,
    allow_dangerous_requests=True,  # Required for newer LangChain versions
)

print("✓ Cypher chain created successfully!")
"""

# ============================================================================
# Alternative: Use separate LLMs for Cypher generation and QA
# ============================================================================

"""
# If you want different models for Cypher generation vs QA:
cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    qa_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    graph=graph,
    verbose=True,
    validate_cypher=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    top_k=10,
    allow_dangerous_requests=True,
)

print("✓ Cypher chain created successfully!")
"""

# ============================================================================
# Test the chain
# ============================================================================

"""
# Test with a simple question
test_question = "What medications are used to treat cardiovascular complications?"

print(f"Question: {test_question}")
print("="*80)

response = cypher_chain.invoke({"query": test_question})
print(f"\nAnswer: {response['result']}")
"""

# ============================================================================
# Test Natural Language to Cypher Query Chain
# Copy these cells into your test_neo4j_FIXED_QUERIES.ipynb notebook
# ============================================================================

# ============================================================================
# CELL 1: Import required libraries
# ============================================================================
"""
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import ChatOpenAI

# Load the cypher generation prompts (now simple variables, not functions!)
import sys
sys.path.append('c:/dev/Diabetes_AgenticGraphRAG/tests')
from cypher_generation_template import cypher_generation_prompt, qa_generation_prompt

print("✓ Imports complete!")
"""

# ============================================================================
# CELL 2: Create the Cypher chain
# ============================================================================
"""
# Create the Cypher chain with your custom prompts
cypher_chain = GraphCypherQAChain.from_llm(
    top_k=10,
    graph=graph,
    verbose=True,
    validate_cypher=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    qa_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    cypher_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

print("✓ Cypher chain created successfully!")
"""

# ============================================================================
# CELL 3: Test with a single question
# ============================================================================
"""
# Test with example questions
test_questions = [
    "What medications are used to treat cardiovascular complications?",
    "What are the A1C target goals for different patient populations?",
    "Which risk factors increase the likelihood of diabetic retinopathy?",
    "What diagnostic tests are used to screen for kidney disease?",
    "What are the recommended treatments for patients with CKD?"
]

# Test the first question
print("Testing question:", test_questions[0])
print("="*80)
response = cypher_chain.invoke({"query": test_questions[0]})
print("\nGenerated Cypher:", response.get('intermediate_steps', 'N/A'))
print("\nAnswer:", response['result'])
"""

# ============================================================================
# CELL 4: Test all questions
# ============================================================================
"""
# Test all questions
for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*80}")
    print(f"Question {i}: {question}")
    print(f"{'='*80}")
    try:
        response = cypher_chain.invoke({"query": question})
        print(f"Answer: {response['result']}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
"""

# ============================================================================
# CELL 5: Test custom question
# ============================================================================
"""
# Try your own question!
custom_question = "What are the complications associated with diabetes?"

print(f"\nCustom Question: {custom_question}")
print("="*80)
response = cypher_chain.invoke({"query": custom_question})
print(f"\nAnswer: {response['result']}")
"""

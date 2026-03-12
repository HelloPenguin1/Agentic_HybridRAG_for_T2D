import streamlit as st
import requests
from typing import Generator
import json
import sseclient  # For Server-Sent Events

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="Diabetes Nursing Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://127.0.0.1:8000"
ENABLE_STREAMING = False # Toggle streaming on/off


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_api_health() -> bool:
    """Check if API is running and ready"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def query_api(question: str) -> dict:
    """Send query to API and get response"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": True,
                "message": f"API Error: {response.status_code}",
                "detail": response.text
            }
            
    except requests.exceptions.Timeout:
        return {
            "error": True,
            "message": "Request timed out. The query is taking too long to process."
        }
    except requests.exceptions.RequestException as e:
        return {
            "error": True,
            "message": f"Connection error: {str(e)}"
        }


def query_api_stream(question: str):
    """Stream query response from API (Server-Sent Events)"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query/stream",
            json={"question": question},
            stream=True,
            timeout=120
        )
        
        if response.status_code == 200:
            # Parse SSE stream
            client = sseclient.SSEClient(response)
            
            for event in client.events():
                if event.data:
                    try:
                        data = json.loads(event.data)
                        yield data
                    except json.JSONDecodeError:
                        continue
        else:
            yield {
                "type": "error",
                "message": f"API Error: {response.status_code}"
            }
            
    except Exception as e:
        yield {
            "type": "error",
            "message": f"Streaming error: {str(e)}"
        }


def format_citation(citation: dict) -> str:
    """Format a citation for display"""
    source_type = citation.get('source_type', 'unknown')
    
    # Icon based on source type
    icon = {
        'vector': '📄',
        'graph': '🔗',
        'web': '🌐'
    }.get(source_type, '📌')
    
    return f"{icon} **[{citation['id']}]** {citation['label']}"


def format_router_info(metadata: dict) -> str:
    """Format router decision info"""
    choice = metadata.get('router_choice', 'unknown')
    
    # Icon based on choice
    icon = {
        'graph': '🔗',
        'vector': '📄',
        'both': '🔗📄',
        'real_time': '🌐'
    }.get(choice, '🤖')
    
    return f"{icon} **{choice.upper()}**"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.title("🩺 Diabetes Nursing")
    st.markdown("### Evidence-Based Assistant")
    st.markdown("---")
    
    # API Status
    if check_api_health():
        st.success("✅ System Ready")
        st.session_state.system_ready = True
    else:
        st.error("❌ API Offline")
        st.warning("Start the API server:\n```bash\npython api/main.py\n```")
        st.session_state.system_ready = False
    
    st.markdown("---")
    
    # Chat Controls
    st.subheader("Chat Controls")
    
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    
    # System Info
    with st.expander("ℹ️ System Information"):
        st.markdown("""
        **Knowledge Sources:**
        - 🔗 Neo4j Knowledge Graph
        - 📄 Qdrant Vector Database
        - 🌐 Real-time Web Search
        
        **Agents:**
        1. Router → Routing strategy
        2. Graph Retriever → Structured facts
        3. Vector Retriever → Contextual info
        4. Evidence Gate → Quality check
        5. Synthesizer → Answer generation
        6. Hallucination Grader → Verification
        7. Citation Agent → Traceability
        """)
    
    # Example Questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **Factual:**
        - What is the typical dose of metformin?
        - What are the contraindications for SGLT2 inhibitors?
        
        **Procedural:**
        - How do I teach insulin injection technique?
        - What are best practices for hypoglycemia management?
        
        **Clinical Decision:**
        - When should I hold metformin before a procedure?
        - Patient on metformin has eGFR of 28, what should I do?
        """)
    
    st.markdown("---")
    st.caption(f"API: {API_BASE_URL}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN INTERFACE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.title("🩺 Type 2 Diabetes Nursing Assistant")
st.markdown("*Evidence-based answers powered by agentic GraphRAG*")

# System check
if not st.session_state.system_ready:
    st.error("⚠️ **System not ready.** Please start the API server first.")
    st.info("Run: `python api/main.py` in your terminal")
    st.stop()

# Welcome message
if len(st.session_state.chat_history) == 0:
    st.markdown("""
    ### 👋 Welcome!
    
    I'm your AI-powered nursing assistant for Type 2 Diabetes care. I can help you with:
    
    - 💊 **Medication information** (doses, contraindications, interactions)
    - 📋 **Clinical protocols** (monitoring, patient teaching, procedures)
    - 🚨 **Decision support** (when to hold meds, contraindication checks)
    - 📚 **Evidence-based guidance** (ADA guidelines, clinical best practices)
    
    All answers include **citations** from authoritative sources for traceability.
    
    **Ask me anything about Type 2 Diabetes nursing care!**
    """)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("🔍 Details", expanded=False):
                metadata = message["metadata"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Router Decision:** {format_router_info(metadata)}")
                    st.caption(metadata.get('router_reasoning', 'N/A'))
                    
                with col2:
                    st.markdown(f"**Web Search:** {'✅ Yes' if metadata.get('web_search_used') else '❌ No'}")
                    st.markdown(f"**Hallucination Check:** {metadata.get('hallucination_score', 'N/A')}")
                
                # Agent flow
                st.markdown("**Execution Flow:**")
                agents = metadata.get('agents_executed', [])
                st.caption(" → ".join(agents))
        
        # Show citations
        if message["role"] == "assistant" and "citations" in message and message["citations"]:
            with st.expander(f"📚 References ({len(message['citations'])})", expanded=False):
                for citation in message["citations"]:
                    st.markdown(format_citation(citation))


# Chat input
if prompt := st.chat_input("Ask me about diabetes nursing care...", disabled=not st.session_state.system_ready):
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        # Streaming mode
        if ENABLE_STREAMING:
            status_placeholder = st.empty()
            answer_placeholder = st.empty()
            
            full_answer = ""
            final_data = None
            
            # Stream the response
            for event in query_api_stream(prompt):
                event_type = event.get("type")
                
                # Status updates
                if event_type == "status":
                    agent = event.get("agent", "system")
                    message = event.get("message", "")
                    status_placeholder.caption(f"🔄 {agent}: {message}")
                
                # Answer chunks
                elif event_type == "answer":
                    chunk = event.get("content", "")
                    full_answer += chunk
                    answer_placeholder.markdown(full_answer + "▌")  # Cursor effect
                
                # Complete
                elif event_type == "complete":
                    final_data = event.get("data", {})
                    full_answer = final_data.get("answer", full_answer)
                    status_placeholder.empty()  # Clear status
                    answer_placeholder.markdown(full_answer)  # Remove cursor
                
                # Error
                elif event_type == "error":
                    error_msg = f"❌ **Error:** {event.get('message', 'Unknown error')}"
                    status_placeholder.empty()
                    answer_placeholder.error(error_msg)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    st.stop()
            
            # Process final data if available
            if final_data:
                citations = final_data.get("citations", [])
                metadata = final_data.get("metadata", {})
                
                # Display metadata
                with st.expander("🔍 Details", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Router Decision:** {format_router_info(metadata)}")
                        st.caption(metadata.get('router_reasoning', 'N/A'))
                        
                    with col2:
                        st.markdown(f"**Web Search:** {'✅ Yes' if metadata.get('web_search_used') else '❌ No'}")
                        st.markdown(f"**Hallucination Check:** {metadata.get('hallucination_score', 'N/A')}")
                    
                    # Agent flow
                    st.markdown("**Execution Flow:**")
                    agents = metadata.get('agents_executed', [])
                    st.caption(" → ".join(agents))
                
                # Display citations
                if citations:
                    with st.expander(f"References ({len(citations)})", expanded=False):
                        for citation in citations:
                            st.markdown(format_citation(citation))
                
                # Save to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_answer,
                    "citations": citations,
                    "metadata": metadata
                })
        
        # Non-streaming mode (fallback)
        else:
            with st.spinner("🤔 Thinking..."):
                response_data = query_api(prompt)
            
            # Handle errors
            if response_data.get("error"):
                error_msg = f"❌ **Error:** {response_data.get('message', 'Unknown error')}"
                st.error(error_msg)
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg
                })
            
            # Display successful response
            else:
                answer = response_data.get("answer", "No answer received")
                citations = response_data.get("citations", [])
                metadata = response_data.get("metadata", {})
                
                # Display answer
                st.markdown(answer)
                
                # Display metadata
                with st.expander("🔍 Details", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Router Decision:** {format_router_info(metadata)}")
                        st.caption(metadata.get('router_reasoning', 'N/A'))
                        
                    with col2:
                        st.markdown(f"**Web Search:** {'✅ Yes' if metadata.get('web_search_used') else '❌ No'}")
                        st.markdown(f"**Hallucination Check:** {metadata.get('hallucination_score', 'N/A')}")
                    
                    # Agent flow
                    st.markdown("**Execution Flow:**")
                    agents = metadata.get('agents_executed', [])
                    st.caption(" → ".join(agents))
                
                # Display citations
                if citations:
                    with st.expander(f"📚 References ({len(citations)})", expanded=False):
                        for citation in citations:
                            st.markdown(format_citation(citation))
                
                # Save to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "metadata": metadata
                })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "🩺 Diabetes Nursing GraphRAG | Powered by Neo4j, Qdrant & LangGraph | "
    f"Messages: {len(st.session_state.chat_history)}"
    "</div>",
    unsafe_allow_html=True
)
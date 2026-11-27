

REVISIONS I AM MAKING IN ORDER:

A - Determined FlashRank compressor actually hurts performance and is not needed. REMOVE from final workflow

B - Tested Query Rewriter under 4 prompt versions , did not work - must add to experiment table

C - CHANGE NODE AND RELATIONSHIP NUMBERS IN THE PAPER --- URGENT    

D = COMPLETELU evaluated graph chain using graph_eval_v2.json dataset. 
    
    IMP
        -qa llm in cypher chain is  no longer used. the raw context is now directlu being sent to state instead of going through 
        qa llm chainl The graph component now only retrieves raw graph context and does no LLM work at all besides cypher 
        transalation.
        
        The langsmith dataset and experiment is under 'Graph_Eval_Dataset_v3'

E - Changed some LLM configurations, on which LLM does what task

F - Made a dataset with 40 questions for E2E evals - 15 V, 15 G, 10 Both
 
  -  Ran complete Evals for 6 workflows minus the 'final workflow' which included web search and evidence gate

  ISSUE FOUND  - Citations are collapsing/ not accurate...
                As per Claude: 
                { The vector retriever fetched the right chunks to answer the question, but the citation agent re-retrieved a different set and numbered them arbitrarily. The match rate between cited sources and gold evidence sections is 0/21 (0%) across all numeric citations in the file.

                The root cause is almost certainly that the citation agent is doing a fresh retrieval (or just taking the top-K from the vector DB at that moment) rather than mapping back to the chunks the vector retriever already returned and stored in state.}      
            
             - The citation validity measures format compliance not citation accuracy

             - Fixed : Citation agent is no longer an agent and is more like a formatter
                       Synthesizer enforces citations itself


G - Ran workflow evas for 6 workflows completely and saved aggregate metrics as CSV
             
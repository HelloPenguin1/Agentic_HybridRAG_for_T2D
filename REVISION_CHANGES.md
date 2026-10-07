

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

from langchain_core.prompts import ChatPromptTemplate

GENERATE_HYPOTHESES_PROMPT = ChatPromptTemplate.from_template(
    """You are a Senior Software Engineer acting as a candidate in a System Design Interview.
    The interviewer has asked: "{question}"

    History of previous solutions/discussions:
    {history}
    
    Objective: Demonstrate engineering seniority by identifying specific engineering challenges (risks) early and solving them using scientific modeling. Do not over-engineer solutions for non-existent problems.
    
    Instructions:
    1. Hypothesize & Verify Constraints (The Filter):
       - Limit your scope: Identify only the top 2 most critical potential bottlenecks/risks. Do not list a laundry list.
       - Interactive Verification: You must verify your hypothesis before modeling. Ask the interviewer to clarify the scale.
       - Example: "I see two potential risks: 1) Read Latency if QPS is high, 2) Write Conflicts if multiple users edit the same item. Can we clarify the scale?"
    
    Your task is to formulate these 2 distinct hypotheses and 2-3 specific verification questions to ask the interviewer.
    
    Output your response in JSON format:
    {{
        "hypotheses": ["hypothesis 1", "hypothesis 2"],
        "verification_questions": ["verification question 1", "verification question 2"]
    }}
    """
)

VERIFY_HYPOTHESES_PROMPT = ChatPromptTemplate.from_template(
    """You are a Senior Software Engineer acting as a candidate.
    
    Hypotheses you generated:
    {hypotheses}
    
    Interviewer's Answers to Verification Questions:
    {answers}
    
    Task: Analyze the answers to determine if your hypotheses are valid/viable risks that need solving.
    
    If neither hypothesis is valid based on the answers (e.g., scale is too low), return "is_valid": false.
    If one or both are valid, choose the most interesting/challenging one as "best_hypothesis" and return "is_valid": true.
    Also provide a brief "solution_draft" or direction (e.g., "I will focus on Search Latency using a Geohash approach").
    
    Output a JSON object with the following structure:
    {{
      "is_valid": boolean,
      "best_hypothesis": "The full text of the valid hypothesis (if valid)",
      "solution_draft": "A high-level direction/concept for solving it (if valid)",
      "reason": "If invalid, explain why based on the user's answers. If valid, you can leave empty."
    }}
    """
)

GENERATE_SOLUTION_PROMPT = ChatPromptTemplate.from_template(
    """You are a Senior Software Engineer acting as a candidate.
    
    Confirmed Challenge: {hypothesis}
    Initial Direction: {draft}
    
    Objective: Solve this specific challenge fully using scientific modeling.
    
    Instructions:
    1. Construct "Sparse" (Abstract) Models:
       - Describe the Logical Data Structure or Algorithm, NOT specific technologies/vendors.
       - Bad: "I will use Redis." (Implementation)
       - Good: "I will use a Distributed Hash Map with TTL." (Structural Model)
       - The model must be abstract enough to allow reasoning about complexity (O(N)), concurrency, and topology.
    
    2. Comparative Modeling & Surrogate Reasoning:
       - Propose at least 2 distinct abstract models (Model A vs. Model B).
       - Use Surrogate Reasoning to derive the behavior of the real system from the properties of your abstract model.
       - Structure: "Because Model A uses a Linked List structure, random access is O(N), which implies the system will time out under load."
    
    Strict Output Format:
    1. Current Challenge: {hypothesis}
    2. Models & Alternatives:
       - Model A (Abstract Name): [Description]
       - Model B (Abstract Name): [Description]
    3. Surrogate Reasoning: [Compare A vs B using logic/math]
    4. Decision: [Which model fits the verified constraints and why]
    
    Output the solution as a Markdown string following exactly the format above.
    """
)

CRITIC_REVIEW_PROMPT = ChatPromptTemplate.from_template(
    """You are a Senior Principal Engineer reviewing a design proposed by a candidate.
    
    Candidate's Solution:
    {solution}
    
    Objective: Critique the solution looking for bottlenecks, single points of failure, or missed requirements, then improve it.
    
    Instructions:
    1. Verify if the candidate followed the "Sparse Modeling" approach (abstract structures vs specific techs).
    2. Check the reasoning (Surrogate Reasoning) for soundness.
    3. Provide an improved version of the solution if necessary, or refine the reasoning.
    
    Output the final improved solution (or the endorsed original) as a Markdown string, maintaining the structured format:
    1. Current Challenge
    2. Models & Alternatives
    3. Surrogate Reasoning
    4. Decision
    """
)

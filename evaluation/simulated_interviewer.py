import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class SimulatedInterviewer:
    def __init__(self, llm):
        self.llm = llm

    def answer_verification(self, questions: List[str], context: str) -> List[str]:
        """Generates answers to verification questions based on the provided context."""
        prompt = ChatPromptTemplate.from_template(
            """You are a System Design Interviewer.
            
            Context about the system we are designing:
            {context}
            
            The candidate has asked the following verification questions:
            {questions}
            
            Please answer these questions based strictly on the context. If the context doesn't specify, invent a reasonable answer that fits the scale.
            Provide answers as a numbered list corresponding to the questions. Make the answer as consice as possible, don't do candidates job by calculating some metrics for them. 
            """
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "questions": "\n".join(questions)})
        return [response] 
        
    def generate_challenge(self, context: str) -> str:
        """Generates a 'what if' challenge based on the new context."""
        prompt = ChatPromptTemplate.from_template(
            """You are a System Design Interviewer.
            
            We are moving to the second phase of the interview.
            New Context/Requirements:
            {context}
            
            Please formulate a short "What if" challenge statement to the candidate to provoke them to adapt their design.
            Example: "Now imagine we need to scale to 1B users. How does this change your design?"
            """
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context})

    def score_report(self, report: str, ideal_outcome: str) -> dict:
        """Scores the final report against the ideal outcome."""
        prompt = ChatPromptTemplate.from_template(
            """You are a System Design Interview Evaluator.
            
            Final Report from Candidate:
            {report}
            
            Ideal Outcome clues:
            {ideal_outcome}
            
            Evaluate the report.
            1. Does it cover the key constraints?
            2. Did it adapt to the second phase?
            3. Are the solutions scientifically sound (metrics backed)?
            
            Output a JSON object with:
            - "reasoning": string with detailed explanation of the provided score
            - "score": integer 0-5:
              0 - candidate failed to cover the key constraints, they don't understand basic system design principles
              1 - candidate managed to understand the task but failed to provide any viable hypotheses
              2 - candidate managed to provide more or less viable hypotheses but their design was very weak and not to the point
              3 - candidate managed to provide good hypotheses and their design was on right track but lacked depth and metrics
              4 - candidate managed to provide good hypotheses and their design was mostly correct backed by good reasoning but lacked depth
              5 - very good hypotheses and the design was correct, backed by good reasoning and with enough depth
            """
        )
        # We need structured output here for the CSV
        from pydantic import BaseModel, Field
        
        class Score(BaseModel):
            score: int = Field(description="Score from 0 to 5")
            reasoning: str = Field(description="Explanation for the score")
            
        structured_llm = self.llm.with_structured_output(Score)
        chain = prompt | structured_llm
        result = chain.invoke({"report": report, "ideal_outcome": ideal_outcome})
        return {"score": result.score, "reasoning": result.reasoning}

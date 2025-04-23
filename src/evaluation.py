from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Create a judge for evaluation
def get_judge_llm(model="gpt-4o-mini"):
    """Get the LLM for evaluation"""
    return ChatOpenAI(model=model, temperature=0)

# Define evaluation prompt
evaluation_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are an expert evaluator tasked with judging the quality of responses to research questions. 
    
Score each response on a scale of 1-10 based on:
1. Accuracy: How factually correct is the information?
2. Relevance: How well does it address the specific question asked?
3. Comprehensiveness: How complete is the coverage of relevant aspects?
4. Evidence: How well supported are the claims with references?
5. Clarity: How clearly is the information presented?

For each criterion, provide a score and a brief justification. Then provide an overall score and summary evaluation.

Your evaluation should follow this format:
Accuracy: [Score] - [Justification]
Relevance: [Score] - [Justification]
Comprehensiveness: [Score] - [Justification]
Evidence: [Score] - [Justification]
Clarity: [Score] - [Justification]

Overall Score: [Average of above scores]

Summary Evaluation: [Overall assessment with key strengths and weaknesses]
"""),
    HumanMessage(content="""
Query: {query}

Response to evaluate: {response}

Please evaluate this response based on the criteria above.
""")
])

def evaluate_response(query, response, model="gpt-4o-mini"):
    """Evaluate a response using the judge LLM"""
    judge_llm = get_judge_llm(model)
    
    evaluation = judge_llm.invoke(
        evaluation_prompt.format(
            query=query,
            response=response
        )
    )
    
    return evaluation.content

def compare_evaluations(query, openai_response, deepseek_response):
    """Compare evaluations of responses from different models"""
    openai_eval = evaluate_response(query, openai_response)
    deepseek_eval = evaluate_response(query, deepseek_response)
    
    # Use a meta-evaluator to compare the responses
    meta_eval_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a meta-evaluator comparing two different AI research assistants.
Based on their responses to the same query and their individual evaluations, determine which system performed better and why.
Provide a concise analysis of the strengths and weaknesses of each system."""),
        HumanMessage(content=f"""
Query: {query}

OpenAI Response:
{openai_response}

OpenAI Evaluation:
{openai_eval}

DeepSeek Response:
{deepseek_response}

DeepSeek Evaluation:
{deepseek_eval}

Which system performed better for this query and why?
""")
    ])
    
    judge_llm = get_judge_llm()
    meta_evaluation = judge_llm.invoke(meta_eval_prompt.format(
            query=query,
            openai_response = openai_response,
            openai_eval = openai_eval,
            deepseek_response = deepseek_response,
            deepseek_eval = deepseek_eval
        ))
    
    return {
        "query": query,
        "openai_evaluation": openai_eval,
        "deepseek_evaluation": deepseek_eval,
        "meta_evaluation": meta_evaluation.content
    }
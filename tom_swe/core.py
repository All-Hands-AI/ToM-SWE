"""Core functionality for ToM-SWE."""

def process_code(code_snippet):
    """
    Process a code snippet using ToM-SWE.
    
    Args:
        code_snippet (str): The code snippet to process.
        
    Returns:
        dict: The processed results.
    """
    # There's a bug here - we're not handling empty code snippets correctly
    if code_snippet == "":
        return {"error": "Code snippet cannot be empty"}
    
    # Process the code snippet
    result = {
        "length": len(code_snippet),
        "lines": code_snippet.count("\n") + 1,
        "analysis": "Sample analysis"
    }
    
    return result
# -------------------------------------------------------------------------------------------------
# Code Like a Pythonista: Idiomatic Python

# Coding style, whitespaces, naming, docstrings, ...
# GOTO: http://python.net/~goodger/projects/pycon/2007/idiomatic/handout.html

# -------------------------------------------------------------------------------------------------
# SVN guidelines: 
#    FIX, DOC, OTH ??? 

# -------------------------------------------------------------------------------------------------
# DOC guidelines:

# Function documentation:
def function_name(input1_name=None, input2_name=None, config1_name=1000.):

    """

    Function description.

    Input:
        input1_name (type): description.
        
        input2_name (type): description.
        
        config1_name (type): description.

    Output:
        output1_name (type): description.
       
        output2_name (type): description.
        
        output3_name (type): description.

    Configurable fields:{"name": "module.function_name", "config": {"config1_name": "1000."}, "inputs": ["input1_name", "input2_name"], "outputs": ["output1_name", "output2_name", "output3_name"]}

    See Also:
        List of related modules. Example:
        module_x
        module_y
        
    Notes:
        Relevant module notes. Example:
        in configurable fields above,
        "config" stands for arguments with default values (e.g., 1000.), 
        "inputs" stand for args with no default values (i.e., None).

    Example:
        An example demonstrating how to use the function.

    References:
        .. [1]    ...
        .. [2]    ...
        .. [3]    ...
    """    
    # Check inputs
    # ...
    # Function Body
    # ...
    # Output
    res = {'output1': None, 'output2': None, 'output3': None}
    return res

# -------------------------------------------------------------------------------------------------
# On top of each module should be:

"""
.. module:: module name                    (e.g., misc)
   :platform: supported platforms         (e.g., Unix, Windows)
   :synopsis: module brief description    (e.g., This module provides several miscellaneous tools.)

.. moduleauthor:: module author            (e.g., Filipe Canento)


"""

# -------------------------------------------------------------------------------------------------
# On bottom of each module should be:
#     unitesting for all functions/classes

if __name__=='__main__':
    import unittest    

    class test(unittest.TestCase):
        """
        A test class for the ...
            """
        def setUp(self):
            # Init
            pass
            # ...
        def test_function_name1(self):
            pass
            # ...
        def test_function_name2(self):
            pass
            # ...            
        # ...
    # Example:
    
    # Unitest:
    unittest.main()    
# -------------------------------------------------------------------------------------------------    

# -------------------------------------------------------------------------------------------------
# Agile Development tips
# Agile methods break development down into short iterations, 
# typically no more than two weeks long, and often as short as a single day.
# Steps:
# 1. Stand-up meeting:
#    a) What did you do yesterday?
#    b) What are you going to do today?
#    c) What is blocking you?
# 2. Pair programming
# 3. Test-driven development: practice of writing unit tests before writing application code.
# 4. Continuous integration: every few minutes, or every time someone commits code to 
#                            the version control repository, an automated process checks 
#                            out a clean copy of the code, builds it, runs all the tests, 
#                            and posts the results somewhere, such as a web page.


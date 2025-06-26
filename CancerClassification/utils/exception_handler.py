import sys
class ExceptionHandler(Exception):
    """
    Custom exception class to help debug faster.
    The class will provide information about - error message, file name, and line number where the exception occured
    """

    def __init__(self, error_message:str, error_details:sys):
        """
        Initialize the class instance

        Args: 
            error_message (str) : Error message describng what went wrong
            error_details (sys) : Details of error extracted by sys to provide more information

        Attributes:
            error_message (str) : Stores the original error
            lineno (int)        : The line number where the error occured
            file_name (str)     : The name of the file where the error occured
        """

        self.error_message = error_message
        _,_,dets = error_details.exc_info()

        self.lineno = dets.tb_lineno
        self.file_name = dets.tb_frame.f_code.co_filename

    def __str__(self):
        """
        Creating a custom layout to print the error message
        """
        return f"Oh no, An error occured in {self.file_name} at line {self.lineno} with the error message - {self.error_message}"
    
if __name__ == "__main__":
    # Testing the intended behavior
    try:
        a = 1/0
    except Exception as e:
        print(ExceptionHandler(e, sys))

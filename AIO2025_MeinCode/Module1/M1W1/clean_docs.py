"""
Module: text
    This module defines classes representing birds and their flying behavior.
    It demonstrates method overriding and the Liskov Substitution Principle (LSP) violation
    by showing how an Ostrich, which cannot fly, overrides the fly method of its parent Bird class.
    Classes:
        Bird: Represents a generic bird with a fly method.
        Ostrich: Represents an ostrich, a bird that cannot fly, overriding the fly method.

Functions:
    make_it_fly(bird: Bird): Calls the fly method on a given Bird instance.
"""

class Bird: # pylint: disable=too-few-public-methods
    """
    Represents a generic bird.

    Methods:
        fly(): Prints a message indicating the bird is flying.
    """
    def fly(self):
        """Prints a message indicating the bird is flying."""
        print("Flying")

class Ostrich(Bird): # pylint: disable=too-few-public-methods
    """
    Represents an ostrich, a type of bird that cannot fly.

    Methods:
        fly(): Prints a message indicating that ostriches cannot fly.
    """
    def fly(self):
        """Raises an error because ostriches can't fly."""
        raise ValueError("Ostriches can't fly")


def make_it_fly(bird: Bird):
    """Calls the fly method on a given Bird instance."""
    bird.fly()

make_it_fly(Ostrich())  # 💥 Error!

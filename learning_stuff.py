# Type Hinting

def print_list(a: list) -> None:
    print(a)

print_list([1, 2, 3])

# Annotates 'radius' to be a float
radius: float = 1.5

# We can annotate a variable without assigning a value!
sample: int

# Annotates 'area' to return a float
def area(r: float) -> float:
    return 3.1415 * r * r


print(area(radius))

# Print all annotations of the function using
# the '__annotations__' dictionary
print('Dictionary of Annotations for area():', area.__annotations__)

from typing import List

# Vector is a list of float values
Vector = List[float]

def scale(scalar: float, vector: Vector) -> Vector:
    return [scalar * num for num in vector]

a = scale(scalar=2.0, vector=[1.0, 2.0, 3.0])
print(a)

from typing import NewType

# Create a new user type called 'StudentID' that consists of
# an integer
StudentID = NewType('StudentID', int)
sample_id = StudentID(100)


from typing import NewType

# Create a new user type called 'StudentID'
StudentID = NewType('StudentID', int)

def get_student_name(stud_id: StudentID) -> str:
    return str(input(f'Enter username for ID #{stud_id}:\n'))

stud_a = get_student_name(StudentID(100))
print(stud_a)

# This is incorrect!!
stud_b = get_student_name(-1)
print(stud_b)


from typing import Any

def print_list(a: Any) -> None:
    print(a)

print_list([1, 2, 3])
print_list(1)

'''
from collections.abc import Callable

def feeder(get_next_item: Callable[[], str]) -> None:
    # Body
    pass

def async_query(on_success: Callable[[int], None],
                on_error: Callable[[int, Exception], None]) -> None:
    # Body
    pass
'''
#
# dataclasses
# This module provides a decorator and functions for automatically adding generated special methods such as __init__() and __repr__() to user-defined classes.

from dataclasses import dataclass

@dataclass
class InventoryItem:
    """Class for keeping track of an item in inventory."""
    name: str
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand
    
    
 
 @dataclass
class Person:
    first_name: str = "Ahmed"
    last_name: str = "Besbes"
    age: int = 30
    job: str = "Data Scientist"

    def __repr__(self):
        return f"{self.first_name} {self.last_name} ({self.age})"

ahmed = Person()
print(ahmed)
# Ahmed Besbes (30)


from dataclass import astuple, asdict

ahmed = Person()
print(astuple(ahmed))
# ('Ahmed', 'Besbes', 30, 'Data Scientist')

print(asdict(ahmed))
# {'first_name': 'Ahmed',
# 'last_name': 'Besbes',
# 'age': 30,
# 'job': 'Data Scientist'}
 
# Frozen instances / immutable objects 
 
@dataclass(frozen=True)
class Person:
     first_name: str = "Ahmed"
     last_name: str = "Besbes"
     age: int = 30
     job: str = "Data Scientist"
 
 
 
 
 
 
 
 
import fnmatch
import os

for file in os.listdir('.'):
    if fnmatch.fnmatch(file, '*.txt'):
        print(file)   
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Python3 solutions and test cases for Cracking the Coding Interview, 6th Edition

Python3.9.5 was used. -- SpiritFryer
"""

__author__ = 'SpiritFryer'
__credits__ = ['McDowell, Gayle Laakmann (2015). Cracking the Coding Interview: 189 Programming Questions and Solutions. CareerCup, LLC. 978-0-9847828-5-7 (ISBN 13)']

# ctci.py
# Cracking the Coding Interview, 6th Edition
# Started: 2021-11-09
# https://github.com/SpiritFryer
# 
# Also see:
# https://github.com/careercup/CtCI-6th-Edition-Python/

# TODO: Challenge: implement streamable/chunkable solutions to all of these. I.e. imagine an extremely large input. Re-write solutions to be able to work by streaming/chunking input/output to work around memory limitations.
# TODO: Use a logging library for debugging, instead of 'print'

################################################
# Imports and general-use functions
################################################
import itertools
import sys
import copy
import types
from typing import Any, Iterable, Iterator, Generator, Tuple, List
from sys import getsizeof


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
    
def zip_(*iterables: Iterable) -> Iterator[Tuple]:  # This can replace the below calls to zip()
  iters = tuple(iter(iterable_) for iterable_ in iterables)
  while True:
    try:
      yield tuple(next(iter_) for iter_ in iters)
    except (StopIteration, RuntimeError) as e:
      break

################################################
# Ch01 Array and Strings
################################################
def is_unique(s: str) -> bool:
  """1.1. Is Unique: Implement an algorithm to determine if a string has all unique characters.

  What if you cannot use additional data structures?
  """
  # set(s): O(n)
  # len(s): O(n), but we can track it while building set(s)
  # len(set(s)): O(n), but we can track it while building set(s)
  #
  # Time: O(n)
  # Space: O(n), but likely much less
  
  #print(f"{getsizeof(set(s)):10} '{s if len(s)<100 else 'String of length:'+str(len(s))}'")
  return len(set(s)) == len(s)

def is_unique_better(s: str) -> bool:
  """1.1. Is Unique: Implement an algorithm to determine if a string has all unique characters.

  What if you cannot use additional data structures?
  """
  # Time: O(k), where k is the size of the character set of s
  # Space: O(k)
  
  seen_chars = set()
  for char_ in s:
    if char_ in seen_chars:
      #print(f"{getsizeof(seen_chars):10} '{s if len(s)<100 else 'String of length:'+str(len(s))}'")
      return False
    else:
      seen_chars.add(char_)
  #print(f"{getsizeof(seen_chars):10} '{s if len(s)<100 else 'String of length:'+str(len(s))}'")
  return True

def is_unique_bit_vector(s: str) -> bool:
  """1.1. Is Unique: Implement an algorithm to determine if a string has all unique characters.

  What if you cannot use additional data structures?
  """
  # Time: O(k), where k is the size of the character set of s
  # Space: O(k)

  seen_chars_bit_vector = 0
  for char_ in s:
    bit_num = ord(char_)
    if (seen_chars_bit_vector >> bit_num) & 1:  # Test bit #bit_num
      #print(f"{getsizeof(seen_chars_bit_vector):10} '{s if len(s)<100 else 'String of length:'+str(len(s))}'")
      return False
    else:
      seen_chars_bit_vector |= 1<<bit_num  # Set bit #bit_num
  #print(f"{getsizeof(seen_chars_bit_vector):10} '{s if len(s)<100 else 'String of length:'+str(len(s))}'")
  return True
  
def is_unique_bit_vector_arbitrary_charsets(s: List[int]) -> bool:
  """1.1. Is Unique: Implement an algorithm to determine if a string has all unique characters.

  In this implementation, s is a list of ints. Each element represents a character in an arbitrary character set.
  """
  # Time: O(k), where k is the size of the character set of s
  # Space: O(k)

  seen_chars_bit_vector = 0
  for bit_num in s:
    if (seen_chars_bit_vector >> bit_num) & 1:  # Test bit #bit_num
      #print(f'{getsizeof(seen_chars_bit_vector):10} {s}')
      return False
    else:
      seen_chars_bit_vector |= 1<<bit_num  # Set bit #bit_num
  #print(f'{getsizeof(seen_chars_bit_vector):10} {s}')
  return True

def is_unique_no_data_structures(s: str) -> bool:
  """1.1. Is Unique: Implement an algorithm to determine if a string has all unique characters.

  What if you cannot use additional data structures?
  """
  # For each char, search whole string to see if it occurs more than once
  # Time: O(n**2)
  # Space: O(1)

  for char_ in s:
    found = False
    for char_check in s:
      if char_check == char_:
        if found:
          return False
        else:  # found == False
          found = True
  else:  # Did not break (or return)
    return True


def check_permutation(s1: str, s2: str) -> bool:
  """1.2. Check Permutation: Given two strings, write a method to decide if one is a permutation of the other."""
  # Space: O(n + m)
  # Time: O(n + m) 
  #       O(n + m) for building the dicts, O(n + m) for comparing them (though we can break as soon as there is a mismatch, during comparison)
  counts_s1 = dict()
  counts_s2 = dict()

  for s, counts in (
      (s1, counts_s1),
      (s2, counts_s2)):
    for char_ in s:
      if char_ in counts:
        counts[char_] += 1
      else:  # char_ not in counts:
        counts[char_] = 1

  return counts_s1 == counts_s2
  
def check_permutation_better(s1: str, s2: str) -> bool:
  """1.2. Check Permutation: Given two strings, write a method to decide if one is a permutation of the other."""
  # Space: O(n + m)
  # Time: O(n + m) 
  #       O(n + m) for building the dicts, O(n + m) for comparing them (though we can break as soon as there is a mismatch, during comparison)
  if len(s1) != len(s2):
    return False
  
  counts = dict()
  for char_ in s1:
    if char_ in counts:
      counts[char_] += 1
    else:  # char_ not in counts:
      counts[char_] = 1
      
  for char_ in s2:
    if char_ not in counts:
      return False
    
    if counts[char_] == 0:
      return False
    counts[char_] -= 1

  return True

def check_permutation_sort(s1: str, s2: str) -> bool:
  """1.2. Check Permutation: Given two strings, write a method to decide if one is a permutation of the other."""
  # Space: O(1), assuming the sorting algo sorts in-place. But since strings are immutable, it's actually O(n + m)
  # Time: O(n log n + m log m) 
  #       O(n log n + m m log m) for sorting, O(min(n, m)) for comparing them
  return sorted(s1) == sorted(s2)


def urlify(s: str) -> str:
  """1.3. URLify: Write a method to replace all spaces in a string with '%20.

  You may assume that the string has sufficient space at the end to hold the additional characters,
  and that you are given the "true" length of the string.
  (Note: If implementing in Java, please use a character array so that you can perform this operation in place.)

  EXAMPLE
    Input: "Mr John Smith "J 13
    Output: "Mr%20J ohn%20Smith"
  """
  # Space: O(1)
  # Time: O(n)
  return s.replace(' ', '%20')

def urlify_manual(s: str) -> str:
  """1.3. URLify: Write a method to replace all spaces in a string with '%20.

  You may assume that the string has sufficient space at the end to hold the additional characters,
  and that you are given the "true" length of the string.
  (Note: If implementing in Java, please use a character array so that you can perform this operation in place.)

  EXAMPLE
    Input: "Mr John Smith "J 13
    Output: "Mr%20J ohn%20Smith"
  """
  # Space: O(n) but only because we are taking in a str which are immutable. If we take in a list, we can do this in-place so technically O(1).
  # Time: O(n)
  output = ''
  for char_ in s:
    if char_ == ' ':
      output += '%20'
    else:
      output += char_
  return output

def urlify_manual_better(s: str) -> str:
  """1.3. URLify: Write a method to replace all spaces in a string with '%20.

  You may assume that the string has sufficient space at the end to hold the additional characters,
  and that you are given the "true" length of the string.
  (Note: If implementing in Java, please use a character array so that you can perform this operation in place.)

  EXAMPLE
    Input: "Mr John Smith "J 13
    Output: "Mr%20J ohn%20Smith"
  """
  # Space: O(n) but only because we are taking in a str which are immutable. If we take in a list, we can do this in-place so technically O(1).
  # Time: O(n)
  return ''.join(char_ if char_ != ' ' else '%20'  for char_ in s)

def urlify_manual_immutable(s: str) -> str:
  """1.3. URLify: Write a method to replace all spaces in a string with '%20.

  You may assume that the string has sufficient space at the end to hold the additional characters,
  and that you are given the "true" length of the string.
  (Note: If implementing in Java, please use a character array so that you can perform this operation in place.)

  EXAMPLE
    Input: "Mr John Smith "J 13
    Output: "Mr%20J ohn%20Smith"
  """
  # Space: O(n)
  # Time: O(n)
  count_spaces = 0
  for char_ in s:
    if char_ == ' ':
      count_spaces += 1
      
  new_len = len(s) + count_spaces*2
  new_s = [''] * new_len
  
  i = len(s) - 1
  new_i = new_len - 1
  
  while i >= 0:
    char_ = s[i]
    
    if char_ == ' ':
      new_s[new_i-2] = '%'
      new_s[new_i-1] = '2'
      new_s[new_i] = '0'
      new_i -= 3
    else:
      new_s[new_i] = s[i]
      new_i -= 1
    
    i -= 1

  return ''.join(new_s)

def palindrome_permutation(s: str) -> bool:
  """1.4. Palindrome Permutation: Given a string, write a function to check if it is a permutation of a palindrome.

  A palindrome is a word or phrase that is the same forwards and backwards. A permutation is a rearrangement of letters.
  The palindrome does not need to be limited to just dictionary words.

  EXAMPLE
    Input: Tact Coa
    Output: True (permutations: "taco cat". "atco cta". etc.)
  """
  # Space: O(n)
  # Time: O(n)

  # At most 1 char could show up an odd number of times. All other chars must show up an even number of times.
  parity = dict()  # parity[char_] == True: even number of times; parity[char_] == False: odd number of times.
  count_odd = 0

  s_condensed = ''.join(s.lower().split())  # Remove all whitespace, ignore case -- O(n), but we could technically just add extra logic to ignore whitespace and convert to lower during the below loop

  for char_ in s_condensed:
    if char_ in parity:
      parity[char_] = not parity[char_]
      if parity[char_]:  # parity[char_] == True: even number of times;
        count_odd -= 1
      else:
        count_odd += 1
    else:
      parity[char_] = False
      count_odd += 1

  return count_odd <= 1


def one_away(s1: str, s2: str) -> bool:
  """1.5. One Away: Given two strings, write a function to check if they are one edit (or zero edits) away.

  There are three types of edits that can be performed on strings:
    insert a character,
    remove a character,
    or replace a character.

  EXAMPLE
    pale, ple -> true
    pales, pale -> true
    pale, bale -> true
    pale, bake -> false
  """
  # Space: O(n + m)
  # Time: O(n + m)

  # Count chars in both. We fail if:
  #   0. len diff > 1  # Too big of a len diff
  #   1. overall diff > 2  # Can replace a char into another one
  #   2. diff within one char's counts > 1  # Can add/delete 1 char

  if abs(len(s1) - len(s2)) > 1:  # Failure by clause 0
    return False

  counts_s1 = dict()
  counts_s2 = dict()

  for s, counts in (
      (s1, counts_s1),
      (s2, counts_s2)):
    for char_ in s:
      if char_ in counts:
        counts[char_] += 1
      else:  # char_ not in counts:
        counts[char_] = 1

  diff = 0
  for char_ in counts_s1.keys() | counts_s2.keys():  # for char_ in set().union(d.keys(), d2.keys()):
    count_s1 = counts_s1[char_] if char_ in counts_s1 else 0
    count_s2 = counts_s2[char_] if char_ in counts_s2 else 0

    diff_of_curr_char = abs(count_s1 - count_s2)

    if diff_of_curr_char > 1:  # Failure by clause 2
      return False

    diff += diff_of_curr_char
    if diff > 2:  # Failure by clause 1
      return False

  else:  # Did not break (or return)
    return True

def one_away_better(s1: str, s2: str) -> bool:
  """1.5. One Away: Given two strings, write a function to check if they are one edit (or zero edits) away.

  There are three types of edits that can be performed on strings:
    insert a character,
    remove a character,
    or replace a character.

  EXAMPLE
    pale, ple -> true
    pales, pale -> true
    pale, bale -> true
    pale, bake -> false
  """
  # Space: O(1)
  # Time: O(max(n, m))

  if abs(len(s1) - len(s2)) > 1:  # 1 insertion or deletion will not be enough
    return False

  num_edits = 0
  iter_s1 = iter(s1)
  iter_s2 = iter(s2)
  #print(f's1 {s1}, len(s1) {len(s1)}, s2 {s2}, len(s2) {len(s2)}')
  try:
    char1 = next(iter_s1)
    char2 = next(iter_s2)
    #print(f'  char1 {char1}, char2 {char2}')
    while True:
      if char1 != char2:
        #print(f'  char1 {char1} != char2 {char2}')
        if num_edits >= 1:  # MAX_EDITS == 1
          #print('    num_edits >= 1 -- return False')
          return False
        elif len(s1) == len(s2):  # make use of 'replace' edit --> skip this position in both
          #print('    len(s1) == len(s2)')
          char1 = next(iter_s1)
          char2 = next(iter_s2)
        elif len(s1) > len(s2):  # make use of either 'insert' or 'delete' edit --> either way skip position in longer s
          #print('    len(s1) > len(s2)')
          char1 = next(iter_s1)
        else:  # len(s1) < len(s2):
          #print('    len(s1) < len(s2)')
          char2 = next(iter_s2)
        num_edits += 1
        #print(f'    num_edits += 1 --> {num_edits}')
      else:  # char1 == char2
        #print(f'  char1 {char1} == char2 {char2}')
        char1 = next(iter_s1)
        char2 = next(iter_s2)
      # No break needed, because next() will raise StopIteration
  except StopIteration:
    #print(f'  StopIteration raised. len(s1) {len(s1)} != len(s2) {len(s2)}: {len(s1) != len(s2)} and num_edits > 0: {num_edits > 0}')
    # if len(s1) != len(s2) and num_edits > 0:
    #   print('  returning False')
    #   return False
    #print('  returning True')
    return True

def string_compression(s: str) -> str:
  """1.6. String Compression: Perform basic string compression using the counts of repeated characters.

  For example, the string aabcccccaaa would become a2b1c5a3.
  If the "compressed" string would not become smaller than the original string, return the original string.

  You can assume the string has only uppercase and lowercase letters (a - z).
  """
  # Space: O(n) but only because we are taking in a str which are immutable. If we take in a list, we can do this in-place so technically O(1).
  # Time: O(n)

  if len(s) <= 1:  # Treating this as an edge-case, as the next part only works for strings of len >= 2
    return s

  new_s = ''
  curr_count = 1
  for curr_char, next_char in pairwise(s):
    if curr_char == next_char:
      curr_count += 1
    else:
      new_s += curr_char + str(curr_count)
      curr_count = 1
  # Finished for, handle last elements
  if curr_char == next_char:  # Did not add curr_char yet, but already counted next_char (the last char)
    new_s += curr_char + str(curr_count)
  else:  # Already added curr_char, and next_char (the last char) is different
    new_s += next_char + '1'

  if len(s) <= len(new_s):
    return s
  else:
    return new_s


def rotate_matrix(m: List[List[Any]]) -> List[List[Any]]:
  """1.7. Rotate Matrix: Rotate an NxN matrix by 90-degrees clockwise.

  Given an image represented by an NxN matrix, where each pixel in the image is 4 bytes, write a method to rotate the
  image by 90 degrees. Can you do this in place?
  """
  # Space: O(n**2) -- this is the non-in-place version
  # Time: O(n**2)
  if len(m) <= 1:
    return m

  new_m = list()
  for new_row in zip(*m):  # zip(*m) is m transposed
    new_m.append(list(reversed(new_row)))  # transposing gives us our rows in the reverse order than we want.
  return new_m

def rotate_matrix_in_place(m: List[List[Any]]) -> List[List[Any]]:
  """1.7. Rotate Matrix: Rotate an NxN matrix by 90-degrees clockwise.

  Given an image represented by an NxN matrix, where each pixel in the image is 4 bytes, write a method to rotate the
  image by 90 degrees. Can you do this in place?
  """
  # Space: O(1) -- this is the in-place version
  # Time: O(n**2)
  if len(m) <= 1:
    return m

  # To rotate in-place, we will rotate successive sequences of 4 pixels in the current "layer".
  # We need to repeat this for all "layers", where a layer is a square of pixels, 1-pixel wide,
  #   and is a certain depth of pixels from the edge of the matrix.

  # At each step, the square layer spans 2-less horizontally and 2-less vertically.
  #   It starts spanning the whole outer square of the matrix.
  num_layers = len(m) // 2

  # [n=3] layer_start: 0; layer_end: 2 | # [n=4] layer_start: 0,1; layer_end: 3,2 | ...  # [layer_start,layer_end]
  #   -- layer_start and layer_end are both inclusive
  for layer_start, layer_end in zip(range(0, num_layers, 1), range(len(m)-1, len(m)-1-num_layers, -1)):
    # [n=3] offset_1: 0,1; offset_2: 2,1 | # [n=4] offset_1: 0,1,2; offset_2: 3,2,1 | ...
    for offset_1, offset_2 in zip(range(layer_start, layer_end, 1), range(layer_end, layer_start, -1)):
      # Rotate 90-degrees clockwise
      #   Start+Offset_1 --> Offset_1+End --> End+Offset_2 --> Offset_2+Start --> Start+Offset_1
      # In other words:
      #   Start+Offset_1 <-- Offset_2+Start <-- End+Offset_2 <-- Offset_1+End <-- Start+Offset_1
      m[layer_start][offset_1], m[offset_2][layer_start], m[layer_end][offset_2], m[offset_1][layer_end] = \
      m[offset_2][layer_start], m[layer_end][offset_2],   m[offset_1][layer_end], m[layer_start][offset_1]

  return m

def zero_matrix(m: List[List[Any]]) -> List[List[Any]]:
  """1.8. Zero Matrix: Modify an MxN matrix such that if an element is 0, its entire row and column are set to 0."""
  # Space: O(m*n)
  # Time: O(m*n*(m+n))
  if len(m) <= 1:
    return m

  num_rows = len(m)
  num_cols = len(m[0])

  new_m = [[x for x in row] for row in m]
  for i,row in enumerate(m):
    set_row_to_zero = False
    for j,x in enumerate(row):
      if x == 0:
        set_row_to_zero = True
        for i_new in range(num_rows):
          new_m[i_new][j] = 0
    if set_row_to_zero:
      for j_new in range(num_cols):
        new_m[i][j_new] = 0
  return new_m

def zero_matrix_better(m: List[List[Any]]) -> List[List[Any]]:
  """1.8. Zero Matrix: Modify an MxN matrix such that if an element is 0, its entire row and column are set to 0."""
  # Space: O(m*n)
  # Time: O(m*n)
  if len(m) <= 1:
    return m

  num_rows = len(m)
  num_cols = len(m[0])

  new_m = [[x for x in row] for row in m]
  cols_to_check = set(range(num_cols))
  #print(f'>>> zero_matrix_better')
  for i, row in enumerate(m):
    #print(f'i {i}, row {row}, cols_to_check {cols_to_check}')
    cols_to_remove = set()
    zero_out_row = False
    for j in cols_to_check:
      #print(f'  j {j}, row[j] {row[j]}')
      if row[j] == 0:
        zero_out_row = True
        #print(f'    row[j] == 0, removing from cols_to_check')
        cols_to_remove.add(j)  # no need to check this column's elements in later iterations
        #print(f'    setting col {j} to 0s in m_new')
        for i_new in range(num_rows):  # set this column in new_m to 0
          new_m[i_new][j] = 0
        #print(f'    setting row {i} to 0s in m_new')
    #print(f'cols_to_check {cols_to_check} = cols_to_check - cols_to_remove {cols_to_remove}  -- {cols_to_check - cols_to_remove}')
    cols_to_check = cols_to_check - cols_to_remove  # set1-set2 is the notation for set difference -- keeping the elements that are in set1 but not in set2.
    if zero_out_row:
      for j_new in cols_to_check:  # set this row in new_m to 0
        new_m[i][j_new] = 0
  return new_m

def zero_matrix_in_place(m: List[List[Any]]) -> List[List[Any]]:
  """1.8. Zero Matrix: Modify an MxN matrix such that if an element is 0, its entire row and column are set to O."""
  # Space: O(m+n)
  # Time: O(m*n)
  if len(m) <= 1:
    return m

  num_rows = len(m)
  num_cols = len(m[0])

  marked_rows = set()
  marked_cols = set()

  # Micro-optimization: could add an edge-case which is when one of the two sets spans the whole matrix, then just need to set the whole thing to 0.
  # This would allow breaking early from set construction, and skip a second (useless) iteration over the matrix.

  for i, row in enumerate(m):
    for j, x in enumerate(row):
      if x == 0:
        marked_rows.add(i)
        marked_cols.add(j)

  for i in marked_rows:
    for j in range(num_cols):
      m[i][j] = 0

  for j in marked_cols:
    for i in range(num_rows):
      m[i][j] = 0

  return m


def is_substring(s1: str, s2: str) -> bool:
  """Checks if s2 is a substring of s1"""
  # Complexity analysis: depends on Python's implementation of str.find -- it is a to-do below to implement this manually.
  # A naive solution is O(m*n)
  # A better solution is O(m+n)
  return s1.find(s2) != -1

def is_substring_manual(s1: str, s2: str) -> bool:
  """Checks if s2 is a substring of s1"""
  pass  # TODO: manually implement is_substring function. -- learn about the algorithms for this

def string_rotation(s1: str, s2: str) -> bool:
  """1.9. String Rotation: Given two strings, s1 and s2, check if s2 is a rotation of s1.

  Assume you have a method isSubstring which checks if one word is a substring of another.
  Solve this using only one call to isSubstring (e.g., "waterbottle" is a rotation of "erbottlewat").
  """
  # Space: O(m), because we concatenate s2 to itself -- (assume mutable)
  # Time: Depends on the implementation of is_substring.
  #       Naive: O(m*n)
  #       Better: O(m+n)

  if (len(s1) == 0 or len(s2) == 0) and len(s1) != len(s2):
    return False
  # if 'erbottlewat' is a rotation of 'waterbottle', then 'waterbottle' is a substring of 'erbottlewaterbottlewat'
  # in other words:
  # if s2 is a rotation of s1, then s1 is a substring of s2+s2
  return is_substring(s2+s2, s1)

def string_rotation_efficient(m: List[List[Any]]) -> List[List[Any]]:
  """1.9. String Rotation: Given two strings, s1 and s2, check if s2 is a rotation of s1.

  Assume you have a method isSubstring which checks if one word is a substring of another.
  Solve this using only one call to isSubstring (e.g., "waterbottle" is a rotation of "erbottlewat").
  """
  pass  # TODO: implement using string search algos

################################################
# Ch02 Linked Lists
################################################
# class base_:
  # x = 'base_x'
  # def basic_print(self):
    # print(f'basic_print x from {self}: {self.x}')
    
# class inherit(base_):
  # # x = 'inherit_x'
  # pass

# x = 'outside_x'
# print('base_ class:', base_.x)
# base_.basic_print(base_)
# print('inherit class:', inherit.x)
# inherit.basic_print(inherit)

# base_obj = base_()
# inherit_obj = inherit()

# base_obj.basic_print()
# inherit_obj.basic_print()
# print('-')

class Node:
  def __init__(self, data=None):
    self.data = data
    self.next_node = None

  def add_node(self, data):  # Todo: This should be in the LinkedList implementation.
    new_node = Node(data)
    if self.next_node is None:  # We are the last node, so new node goes after us.
      self.next_node = new_node
    else:  # We are not the last node, so we need to fit in the new node and link our current next node to it.
      new_node.next_node = self.next_node
      self.next_node = new_node
    return new_node

  def del_next_node(self):
    self.next_node = self.next_node.next_node
    
  def __str__(self):
    return f'{str(self.data)}'


class SinglyLinkedList:
  def __init__(self, data=None):
    if data is None:
      self.first_node = None
      self.last_node = None
      self.num_nodes = 0
    else:
      try:
        iterator = iter(data)  # test if data is iterable -- if it is not it would raise TypeError
        try:  # data is iterable. Test if it has any elements in it
          self.first_node = Node(next(iterator))
          self.last_node = self.first_node
        except StopIteration:  # Empty
          self.first_node = None
          self.last_node = None
          self.num_nodes = 0
          return
        
        # data is iterable and is not empty. Initialize using its contents.
        curr_node = self.first_node
        count = 1
        for count,item in enumerate(iterator,2):
          curr_node = curr_node.add_node(item)
        self.last_node = curr_node
        self.num_nodes = count
      
      except TypeError:  # data is not iterable
        self.first_node = Node(data)
        self.last_node = self.first_node
        self.num_nodes = 1

  def push_start(self, data):
    new_node = Node(data)
    new_node.next_node = self.first_node
    self.first_node = new_node
    if self.last_node is None:
      self.last_node = self.first_node
    self.num_nodes += 1
    return self
    
  def append_end(self, data):
    if self.first_node is None:
      self.first_node = Node(data)
      self.last_node = self.first_node
      self.num_nodes = 1
    else:
      # for node in self.__iter__():
        # if node.next_node is None:
          # node.add_node(data)
          # break
      self.last_node = self.last_node.add_node(data)
      self.num_nodes += 1
    return self
  
  def pop_start(self):
    if self.first_node is None:
      return
    data = self.first_node.data
    self.first_node = self.first_node.next_node
    self.num_nodes -= 1
    return data
    
  def pop_end(self):  # TODO: add second_to_last, to enable O(1) pop_end. Current: O(n)
    if self.first_node is None:
      return
    elif self.first_node.next_node is None:
      data = self.first_node.data
      self.first_node = None
      self.num_nodes -= 1
      return data
    else: 
      for node in self.__iter__():
        if node.next_node.next_node is None:
          data = node.next_node.data
          node.next_node = None
          self.num_nodes -= 1
          return data
    
  def __iter__(self):
    self.curr_iter_steps = 1
    self.curr_iter_node = self.first_node
    return self
    
  def __next__(self):
    if self.curr_iter_node is None or self.curr_iter_steps > self.num_nodes:  # To avoid infinite loops caused by a Node linking to a previous Node.
      raise StopIteration
    node = self.curr_iter_node
    self.curr_iter_node = self.curr_iter_node.next_node
    self.curr_iter_steps += 1
    return node

  def __str__(self):
    return f'[{" -> ".join(str(node.data) for node in self.__iter__())}]'
    
  def __len__(self):
    return self.num_nodes
    
  def __copy__(self):
    new_ll = SinglyLinkedList(node.data for node in self.__iter__())
    return new_ll


# TODO: implement tests for the class and its functions. -- it is not as easy as for the self-contained functions, because some of these functions modify the mutable data. So we need to expand the testing suite such that we can specify how/what to inspect, rather than simply relying on the function's return.
# print('Testing SinglyLinkedList')
# #ll = SinglyLinkedList()
# #ll = SinglyLinkedList(['data_0', 'data_1', 'data_2', 'data_3'])
# # ll = SinglyLinkedList(['data_0'])
# ll = SinglyLinkedList('abcd')

# print('len: ', len(ll))
# print('  start iterating')
# for node in ll:
  # print(node.data)
# print('  done iterating')
# print('ll:', ll)

# # print('ll.push_start:', ll.push_start('data_1'), ll)
# # print('  start iterating')
# # for node in ll:
  # # print(node.data)
# # print('  done iterating')

# # print('ll.append_end:', ll.append_end('data_2'), ll)
# # print('ll.push_start:', ll.push_start('data_0'), ll)
# # print('ll.append_end:', ll.append_end('data_3'), ll)

# print('ll.pop_end:', ll.pop_end(), ll)
# print('ll.pop_start:', ll.pop_start(), ll)
# print('ll.pop_end:', ll.pop_end(), ll)
# print('ll.pop_end:', ll.pop_end(), ll)
# print('ll.pop_start:', ll.pop_start(), ll)
# print('ll.pop_end:', ll.pop_end(), ll)


def remove_dups(ll: SinglyLinkedList) -> str:  # Output string representation of resultant linked list -- for easier testing, so we do not have to implement equality testing of two LinkedList objects. TODO: implement equality testing and return an actual linked list.
  """2.1. Remove Dups: Write code to remove duplicates from an unsorted linked list.
  FOLLOW UP
  How would you solve this problem if a temporary buffer is not allowed?
  """
  # Space: O(n)
  # Time: O(n)
  
  # Copy, so that the test cases can be re-used -- EDIT: Never mind, implemented copying of test cases instead.
  # new_ll = SinglyLinkedList(node.data for node in ll)
  # ll = new_ll
  
  if len(ll) > 0:
    seen = set()
    curr_node = ll.first_node
    seen.add(curr_node.data)
    while True:
      if curr_node.next_node is None:
        break
      next_node = curr_node.next_node
      while next_node.data in seen:
        if next_node.next_node is None:
          curr_node.next_node = None
          break
        next_node = next_node.next_node
      else:  # Did not break
        curr_node.next_node = next_node
        
        curr_node = next_node
        seen.add(curr_node.data)
        continue
      # Did break --> done
      break
  
  return str(ll)

def remove_dups_no_buffer(ll: SinglyLinkedList) -> str:  # Output string representation of resultant linked list -- for easier testing, so we do not have to implement equality testing of two LinkedList objects. TODO: implement equality testing and return an actual linked list.
  """2.1. Remove Dups: Write code to remove duplicates from an unsorted linked list.
  FOLLOW UP
  How would you solve this problem if a temporary buffer is not allowed?
  """
  # Space: O(1)
  # Time: O(n**2), technically O(n**2/2 + n/2), since we do: for i: 0->n-1, for j: 0->i+1. I.e.: 1 + 2 + ... + n  =  n(n+1)/2  = n**2/2 + n/2
  
  # Copy, so that the test cases can be re-used -- EDIT: Never mind, implemented copying of test cases instead.
  # new_ll = SinglyLinkedList(node.data for node in ll)
  # ll = new_ll
  
  #print('remove_dups_no_buffer', ll)
  if len(ll) > 0:
    i = 0
    curr_node = ll.first_node
    while True:
      #print('  ', i, ':', curr_node, '->', curr_node.next_node, 'll:', str(ll))
      if curr_node.next_node is None:
        #print('curr_node.next_node is None')
        break
        
      test_node = ll.first_node
      for j in range(i+1):
        #print('    ', j, ':', test_node)
        
        if curr_node.next_node.data == test_node.data:
          #print('curr_node.next_node.data == test_node.data')
          curr_node.next_node = curr_node.next_node.next_node
          break
        test_node = test_node.next_node
      else:  # Did not break
        i += 1
        curr_node = curr_node.next_node
        if curr_node is None:
          break
  
  #print('Done:', str(ll))
  return str(ll)


def return_kth_to_last(ll: SinglyLinkedList, k: int) -> str:  # Output string representation of resultant linked list -- for easier testing, so we do not have to implement equality testing of two LinkedList objects. TODO: implement equality testing and return an actual linked list.
  """2.2. Return Kth to Last: Implement an algorithm to find the kth to last element of a singly linked list."""
  # In this one, we assume our LinkedList implementation does not keep track of the number of items it has.
  
  # Space: O(1)
  # Time: O(n + k)
  
  n = 0
  # Go through the whole list to find its length
  for n, _ in enumerate(ll, 1):
    pass
  
  # Restart and go n-k steps
  target = n - k  # In a 0-based index, with n elements, target is the n-k'th element. E.g. [1,2,3,4,5], k=1 -> 5-1 = 4th element (0-based index)
  for i, node in enumerate(ll):
    if i == target:
      return node.data
  

def return_kth_to_last_window(ll: SinglyLinkedList, k: int) -> str:  # Output string representation of resultant linked list -- for easier testing, so we do not have to implement equality testing of two LinkedList objects. TODO: implement equality testing and return an actual linked list.
  """2.2. Return Kth to Last: Implement an algorithm to find the kth to last element of a singly linked list."""
  # Space: O(k)
  # Time: O(n)
  
  # In this one, we assume our LinkedList implementation does not keep track of the number of items it has.
  # We keep track of the last k elements, using a queue. Therefore, when we reach the end, we just need to return the oldest saved item.
  
  # For this, a dequeue is more efficient than a list utilized as a queue, because popping from the start of a list forces all elements to be moved.
  # Ironically, a dequeue is a doubly linked list.
  
  # We can get around using the builtin dequeue because in our use case we have a fixed size of items we maintain (k), so we can just use a regular list and overwrite elements rather than popping, and maintain an index to the current "latest".
  q_index = -1
  queue = list()
  q_size = 0
  for i, node in enumerate(ll):
    if q_size < k:
      queue.append(node.data)
      q_size += 1
    else:
      q_index += 1
      if q_index >= k:
        q_index = 0
      queue[q_index] = node.data

  # q_index is where we last inserted, so nth item. One next wraps back to kth item.
  q_index += 1
  if q_index >= k:
    q_index = 0
    
  return queue[q_index]
  
def return_kth_to_last_cheat(ll: SinglyLinkedList, k: int) -> str:  # Output string representation of resultant linked list -- for easier testing, so we do not have to implement equality testing of two LinkedList objects. TODO: implement equality testing and return an actual linked list.
  """2.2. Return Kth to Last: Implement an algorithm to find the kth to last element of a singly linked list."""
  # In this one, we cheat because our LinkedList implementation keeps track of the number of items it has.
  
  # Space: O(1)
  # Time: O(n-k)
  
  target = len(ll) - k  # E.g. k = 2, n = 5, then we want i = 5-2 = 3, where i is a 0-based index. [a -> b -> c -> d -> e] -- here i=3 is d, with a zero-based counting.
  for i, curr_node in enumerate(ll):
    if i == target:
      return curr_node.data


def delete_middle_node(ll: SinglyLinkedList, data: Any) -> str:  # Output string representation of resultant linked list -- for easier testing, so we do not have to implement equality testing of two LinkedList objects. TODO: implement equality testing and return an actual linked list.
  """2.3. Delete Middle Node: Implement an algorithm to delete a node in the middle (i.e., any node but the first and last node, not necessarily the exact middle) of a singly linked list, given only access to that node.
  EXAMPLE
  Input: the node c from the linked list a -> b -> c -> d -> e -> f
  Result: nothing is returned, but the new linked list looks like a -> b -> d -> e -> f
  """
  # Space: O(1)
  # Time: O(n)
  
  curr_node = ll.first_node
  while True:
    if curr_node.next_node is None:
      break
  
    if curr_node.next_node.data is data:
      curr_node.next_node = curr_node.next_node.next_node
    else:
      curr_node = curr_node.next_node
    
  return str(ll)  # For testing purposes
  
def delete_middle_node_object(ll: SinglyLinkedList, node_to_delete: Node) -> str:  # Output string representation of resultant linked list -- for easier testing, so we do not have to implement equality testing of two LinkedList objects. TODO: implement equality testing and return an actual linked list.
  """2.3. Delete Middle Node: Implement an algorithm to delete a node in the middle (i.e., any node but the first and last node, not necessarily the exact middle) of a singly linked list, given only access to that node.
  EXAMPLE
  Input: the node c from the linked list a -> b -> c -> d -> e -> f
  Result: nothing is returned, but the new linked list looks like a -> b -> d -> e -> f
  """
  # Space: O(1)
  # Time: O(n)
  
  curr_node = ll.first_node
  while True:
    if curr_node.next_node is None:
      break
  
    if curr_node.next_node is node_to_delete:
      curr_node.next_node = curr_node.next_node.next_node
    else:
      curr_node = curr_node.next_node
    
  return str(ll)  # For testing purposes


def partition_ll(ll: SinglyLinkedList, pivot: Any) -> str:  # Output string representation of resultant linked list -- for easier testing, so we do not have to implement equality testing of two LinkedList objects. TODO: implement equality testing and return an actual linked list.
  """2.4. Partition: Write code to partition a linked list around a value x, such that all nodes less than x come before all nodes greater than or equal to x. If x is contained within the list, the values of x only need to be after the elements less than x (see below). The partition element x can appear anywhere in the "right partition"; it does not need to appear between the left and right partitions.
  EXAMPLE
  Input: 3 -> 5 -> 8 -> 5 - > 10 -> 2 -> 1 [partition = 5]
  Output: 3 -> 1 -> 2 -> 10 -> 5 -> 5 -> 8
  """
  # Space: O(1)
  # Time: O(n)
  
  # Build two chains: left_chain: elements < pivot, right_chain: elements >= pivot
  #   Keep track of: first and last Node of each chain.
  #   At the end: ll.first_node = left_chain_start,  left_chain_end.next_node = right_chain_start,  right_chain_end.next_node = None
  
  if ll.first_node is None:
    return str(ll)
  
  left_chain_start = left_chain_end = None
  right_chain_start = right_chain_end = None
  
  curr_node = ll.first_node
  while True:
    if curr_node.data < pivot:  # Add curr_node to left_chain
      if left_chain_start is None:
        left_chain_start = left_chain_end = curr_node
      else:
        left_chain_end.next_node = curr_node
        left_chain_end = curr_node
    else:  # Add curr_node to right_chain
      if right_chain_start is None:
        right_chain_start = right_chain_end = curr_node
      else:
        right_chain_end.next_node = curr_node
        right_chain_end = curr_node
        
    if curr_node.next_node is None:
      break
      
    curr_node = curr_node.next_node
  
  if left_chain_start is not None:
    ll.first_node = left_chain_start
    
    if right_chain_start is not None:
      left_chain_end.next_node = right_chain_start
      right_chain_end.next_node = None  
  #else:  #left_chain_start is None:  # There was nothing that's < pivot -- above code essentially did nothing.
  #  pass
  
  return str(ll)  # For testing purposes


def sum_lists(ll_a: SinglyLinkedList, ll_b: SinglyLinkedList) -> str:  # Output string representation of resultant linked list -- for easier testing, so we do not have to implement equality testing of two LinkedList objects. TODO: implement equality testing and return an actual linked list.
  """2.5. Sum Lists: You have two numbers represented by a linked list, where each node contains a single digit. The digits are stored in reverse order, such that the 1 's digit is at the head of the list. Write a function that adds the two numbers and returns the sum as a linked list.
  
  EXAMPLE
  Input: (7 -> 1 -> 6) + (5 -> 9 -> 2). That is, 617 + 295.
  Output: 2 -> 1 -> 9. That is, 912.
  
  FOLLOW UP
  Suppose the digits are stored in forward order. Repeat the above problem.
  
  EXAMPLE
  Input: (6 -> 1 -> 7) + (2 -> 9 -> 5). That is, 617 + 295.
  Output: 9 -> 1 -> 2. That is, 912.
  """
  # Space: O(n)
  # Time: O(n)
  #   where n = max(len(ll_1), len(ll_2))
  
  if ll_a.first_node is None:
    return str(SinglyLinkedList(node.data for node in ll_b))
  if ll_b.first_node is None:
    return str(SinglyLinkedList(node.data for node in ll_a))
  
  carry = 0
  sum_ll = SinglyLinkedList()
  for node_a, node_b in zip(ll_a, ll_b):
    digit_sum = node_a.data + node_b.data + carry
    sum_ll.append_end(digit_sum%10)
    carry = digit_sum // 10
  
  # At this stage, there may be leftover nodes in one of the lls.
  if node_a.next_node is not None or node_b.next_node is not None:
    node = node_a.next_node if node_a.next_node is not None else node_b.next_node
    
    while True:
      digit_sum = node.data + carry
      sum_ll.append_end(digit_sum%10)
      carry = digit_sum // 10
      
      if node.next_node is None:
        break
      node = node.next_node

  # Whether or not the above if ran, either way we need to do this:
  if carry > 0:
    sum_ll.append_end(carry)
    
  return str(sum_ll)  # For testing purposes

def sum_lists_reverse_representation(ll_a: SinglyLinkedList, ll_b: SinglyLinkedList) -> str:  # Output string representation of resultant linked list -- for easier testing, so we do not have to implement equality testing of two LinkedList objects. TODO: implement equality testing and return an actual linked list.
  """2.5. Sum Lists: You have two numbers represented by a linked list, where each node contains a single digit. The digits are stored in reverse order, such that the 1 's digit is at the head of the list. Write a function that adds the two numbers and returns the sum as a linked list.
  
  EXAMPLE
  Input: (7 -> 1 -> 6) + (5 -> 9 -> 2). That is, 617 + 295.
  Output: 2 -> 1 -> 9. That is, 912.
  
  FOLLOW UP
  Suppose the digits are stored in forward order. Repeat the above problem.
  
  EXAMPLE
  Input: (6 -> 1 -> 7) + (2 -> 9 -> 5). That is, 617 + 295.
  Output: 9 -> 1 -> 2. That is, 912.
  """
  # Space: O(n) .. O(2*n + 1 + m)
  # Time: O(n) .. O(2*n + 1 + m)
  #   where n = max(len(ll_1), len(ll_2))
  #
  # Or: Space O(n), Time O(n**2)
  
  if ll_a.first_node is None:
    return str(SinglyLinkedList(node.data for node in ll_b))
  if ll_b.first_node is None:
    return str(SinglyLinkedList(node.data for node in ll_a))
  
  a_arr = list(node.data for node in ll_a)
  b_arr = list(node.data for node in ll_b)
  
  carry = 0
  sum_arr = list()
  for a_data, b_data in zip(reversed(a_arr), reversed(b_arr)):
    digit_sum = a_data + b_data + carry
    sum_arr.append(digit_sum%10)
    carry = digit_sum // 10
  
  # At this stage, there may be leftover nodes in one of the lls.
  len_diff = abs(len(a_arr) - len(b_arr))
  if len_diff > 0:
    arr = a_arr if len(a_arr) > len(b_arr) else b_arr
    
    for data in arr[len_diff-1::-1]:
      digit_sum = data + carry
      sum_arr.append(digit_sum%10)
      carry = digit_sum // 10

  # Whether or not the above if ran, either way we need to do this:
  if carry > 0:
    sum_arr.append(carry)

  return str(SinglyLinkedList(reversed(sum_arr)))  # For testing purposes


def palindrome_ll(ll: SinglyLinkedList) -> bool:
  """2.6. Palindrome: Implement a function to check if a linked list is a palindrome."""
  # Space: O(n)
  # Time: O(n)
  #
  # Or: Space O(1), Time O(n**2)
  
  if ll.first_node is None:
    return True
  
  arr = list(node.data for node in ll)
  
  mid = len(arr) // 2
  for left, right in zip(arr[:mid+1], arr[-1:mid-1:-1]):
    if left != right:
      return False
  return True

def palindrome_ll_manual(ll: SinglyLinkedList) -> bool:
  """2.6. Palindrome: Implement a function to check if a linked list is a palindrome."""
  # Space O(1), Time O(n**2)
  
  if ll.first_node is None:
    return True
  
  mid = len(ll) // 2
  left_node = ll.first_node
  for target_left, target_right in zip(range(0,mid+1), range(len(ll)-1, mid-1, -1)):
    if target_left != 0:
      left_node = left_node.next_node
      
    right_node = left_node
    for _ in range(target_right - target_left):
      right_node = right_node.next_node
      
    if left_node.data != right_node.data:
      return False
  return True


def intersection_ll(ll_a: SinglyLinkedList, ll_b: SinglyLinkedList) -> Node:
  """2.7 Intersection: Given two (singly) linked lists, determine if the two lists intersect. Return the intersecting node.
  Note that the intersection is defined based on reference, not value. That is, if the kth node of the first linked list is the exact same node (by reference) as the jth node of the second linked list, then they are intersecting.
  """
  # Space O(n+m)
  # Time O(n+m)
  
  nodes_a = set()
  nodes_b = set()
  
  node_a = node_b = None
  for node_a, node_b in zip(ll_a, ll_b):
    nodes_a.add(node_a)
    nodes_b.add(node_b)
    
    if node_a in nodes_b:
      return node_a
    if node_b in nodes_a:
      return node_b
      
  # One of the two might have leftovers 
  node_ = None
  nodes = None
  if node_a is not None and node_a.next_node is not None:
    node_ = node_a
    nodes = node_b
  elif node_b is not None and node_b.next_node is not None:
    node_ = node_b
    nodes = node_a
    
  if node_ is not None:
    while True:
      node_ = node_.next_node
      if node_ in nodes:
        return node_
      if node_.next_node is None:
        break
        
  return None

def intersection_ll_manual(ll_a: SinglyLinkedList, ll_b: SinglyLinkedList) -> Node:
  """2.7 Intersection: Given two (singly) linked lists, determine if the two lists intersect. Return the intersecting node.
  Note that the intersection is defined based on reference, not value. That is, if the kth node of the first linked list is the exact same node (by reference) as the jth node of the second linked list, then they are intersecting.
  """
  # Space O(1)
  # Time O(n * m)
  
  for node_a in ll_a:
    for node_b in ll_b:
      if node_a is node_b:
        return node_a

  return None

def ll_generator_wrapper(ll: SinglyLinkedList) -> Generator:
  """Generator function that iterates through a SinglyLinkedList, yields each Node."""
  if not isinstance(ll, SinglyLinkedList):
    raise TypeError
  if ll.first_node is None:
    return None
  
  node = ll.first_node
  yield node
  while node.next_node is not None:
    node = node.next_node
    yield node

def intersection_ll_manual_generator(ll_a: SinglyLinkedList, ll_b: SinglyLinkedList) -> Node:
  """2.7 Intersection: Given two (singly) linked lists, determine if the two lists intersect. Return the intersecting node.
  Note that the intersection is defined based on reference, not value. That is, if the kth node of the first linked list is the exact same node (by reference) as the jth node of the second linked list, then they are intersecting.
  """
  # Space O(1)
  # Time O(n * m)
  
  # for node_a in ll_a:
    # for node_b in ll_b:
      # if node_a is node_b:
        # return node_a
  for node_a in ll_generator_wrapper(ll_a):
    for node_b in ll_generator_wrapper(ll_b):
      if node_a is node_b:
        return node_a

  return None


def ll_loop_detection(ll: SinglyLinkedList) -> Node:
  """2.8 Loop Detection: Given a circular linked list, implement an algorithm that returns the node at the beginning of the loop.
  DEFINITION
    Circular linked list: A (corrupt) linked list in which a node's next pointer points to an earlier node, so as to make a loop in the linked list.
  EXAMPLE
    Input: A -> B -> C -> D -> E -> C [the same C as earlier]
    Output: C
  """
  # Space O(n)
  # Time O(n)

  if ll.first_node is None:
    return None

  nodes_set = {ll.first_node}
  
  for node in ll:
    if node.next_node in nodes_set:
      return node.next_node
    nodes_set.add(node)

  return None

def ll_loop_detection_manual(ll: SinglyLinkedList) -> Node:
  """2.8 Loop Detection: Given a circular linked list, implement an algorithm that returns the node at the beginning of the loop.
  DEFINITION
    Circular linked list: A (corrupt) linked list in which a node's next pointer points to an earlier node, so as to make a loop in the linked list.
  EXAMPLE
    Input: A -> B -> C -> D -> E -> C [the same C as earlier]
    Output: C
  """
  # Space O(n)
  # Time O(n)

  if ll.first_node is None:
    return None

  nodes_set = set()
  node = ll.first_node
  while node.next_node is not None:
    if node.next_node in nodes_set:
      return node.next_node
    nodes_set.add(node)
    node = node.next_node

  return None


################################################
# Ch3
################################################
class StackIsFullError(Exception):
  """Indicates that the stack is full -- an item was attempted to be pushed, but the stack is already at full capacity."""
  pass

def three_in_one(arr: List,
                 bounding_indices: List[int], 
                 request: int, 
                 operate_on: int = None, 
                 push_data: Any = None,
                 _indices = dict()) -> Any:
  """3.1 Three in One: Describe how you could use a single array to implement three stacks.
  
  This function is implemented more or less as a class. `_indices` is like a private variable, containing metadata about the state of a three_in_one stack defined by the arguments (arr, bounding_indices).
  
  `request` indicates the action we are taking. 0: initialize, 1: push, 2: pop, 3: peek.
  
  `operate_on` is a 0-based index, indicating which stack number to perform the operation on.
  
  1: push, requires `operate_on` and `push_data`. -- Pushes `push_data` to stack number `operate_on`.
  2: pop, requires `operate_on` -- Pops the latest item from stack number `operate_on`, returning the item and removing it from this stack.
  3: peek, requires `operate_on` -- Peeks at the latest item from stack number `operate_on`, returning the item while keeping it in the stack.
  """
  
  # Check arguments
  if arr is None:
    raise ValueError(f'Argument `arr` must be a list.')
  
  if bounding_indices is None:
    raise ValueError(f'Argument `bounding_indices` must be a list.')
  
  if len(arr) != bounding_indices[-1]+1:
    raise ValueError(f'Argument `arr` must contain exactly bounding_indices[-1]+1 number of elements. In other words, len(arr) == bounding_indices[-1]+1 must be True.')
  
  for prev_i, curr_i in pairwise(bounding_indices):
    if prev_i >= curr_i:
      raise ValueError(f'bounding_indices must contain indices that are in ascending order. In other words bounding_indices[0] < bounding_indices[1] < ... < bounding_indices[n-1]')
  
  if request < 0 or request > 3:
    raise ValueError(f'Argument `request` must be in the range 0 <= `request <= 3.')
    
  id_ = (id(arr), id(bounding_indices))
    
  # Initialize if needed
  if id_ not in _indices:
    _indices[id_] = [0] + [i+1 for i in bounding_indices[:-1]]

  # Check arguments
  if operate_on is None:
    raise ValueError(f'Argument `operate_on` must be specified for `request` == {request}.')
  if operate_on >= len(bounding_indices):
    raise ValueError(f'Argument `operate_on` must be an index to `bounding_indices`. Therefore `operate_on` must be 0 <= operate_on < len(bounding_indices).')

  #print(arr, bounding_indices, request, operate_on, push_data, _indices, sep='\n  ', end='\n\n\n')  # DEBUG
  # Push logic
  if request == 1:
    # Stack is full
    if _indices[id_][operate_on] > bounding_indices[operate_on]:
      raise StackIsFullError
    # Stack is not full
    else:
      arr[_indices[id_][operate_on]] = push_data
      _indices[id_][operate_on] += 1

  # Pop logic
  elif request == 2:
    # Stack is empty
    if _indices[id_][operate_on] == (0 if operate_on == 0 else bounding_indices[operate_on-1]+1):
      return None
    # Stack is not empty
    else:
      _indices[id_][operate_on] -= 1
      return arr[_indices[id_][operate_on]]

  # Peek logic
  elif request == 3:
    # Stack is empty
    if _indices[id_][operate_on] == (0 if operate_on == 0 else bounding_indices[operate_on-1]+1):
      return None
    # Stack is not empty
    else:
      return arr[_indices[id_][operate_on]-1]


class MyStack:
  pass  # TODO

class StackMin(MyStack):
  pass  # TODO

def stack_min(ll: SinglyLinkedList) -> Node:
  """3.2 Stack Min: How would you design a stack which, in addition to push and pop, has a function min which returns the minimum element? Push, pop and min should all operate in 0(1) time."""
  pass  # TODO


################################################
# Main
################################################
if __name__ == '__main__':
  #sys.stdout = open('output.txt', 'w', encoding='utf-8')  # Re-route stdout to file

  # `checks` (declared later) contains testing info, to test the defined functions with a set of inputs and corresponding expected outputs. Below is a rough schema.
  #   `funcs` is a list of functions you want to test together -- they will all be given the same test cases. Useful when we have different implementations of the same function.
  #   `duplication_flag` is a bool that indicates whether `inputs` should be copied (True) or not (False). This is important to be set to True for functions that modify mutable test cases, when there is more than 1 function version being tested on the same test case. And important to be False for functions that rely on exact object inputs for equality testing.
  #   `require_is_match` is a bool that indicates whether the function's output should be compared to `expected_output` using `is` (True) or `==` (False).
  #   `input_args` are the arguments passed to the function, in the current test case. It is a tuple. Can be empty if there are no input arguments.
  #   `input_kwargs` are the keyword arguments passed to the function, in the current test case. It is a dict. Can be empty if there are no input keyword arguments.
  #   `expected_output` is the output we expect from the function, given the inputs of the current test case
  #
  # checks = (
  #   ((funcs),
  #     (
  #       (duplication_flag, require_is_match, (input_args), {input_kwargs}, expected_output),
  #       ... # more test cases
  #     )
  #   ),
  #   ... # more functions to test
  # )

  # Test cases
  
  ################################################
  # Ch1
  ################################################
  checks_ch1 = (    
    ((is_unique, is_unique_better, is_unique_bit_vector, is_unique_no_data_structures,), 
      (
        (True, False, ('',), dict(), True),
        (True, False, ('a',), dict(), True),
        (True, False, ('aa',), dict(), False),
        (True, False, ('asd',), dict(), True),
        (True, False, ('asda',), dict(), False),
        (True, False, ('😁',), dict(), True),
        (True, False, ('😁😁',), dict(), False),
        (True, False, ('a😁',), dict(), True),
        (True, False, ('😁a😁',), dict(), False),
        (True, False, ('a😁a',), dict(), False),
        (True, False, (''.join(chr(x) for x in range(128)),), dict(), True),  # ASCII
        (True, False, (''.join(chr(x) for x in range(200)),), dict(), True),  # First 200 Unicode
        (True, False, (''.join(chr(x) for x in range(1114111,1114111-200,-1)),), dict(), True),  # Last 200 Unicode
        # ((''.join(chr(x) for x in range(1114111)),), dict(), True),  # All Unicode
      )
    ),
    ((is_unique_bit_vector_arbitrary_charsets,),
      (
        (True, False, ([],), dict(), True),
        (True, False, ([1],), dict(), True),
        (True, False, ([1, 1],), dict(), False),
        (True, False, ([1, 2],), dict(), True),
        (True, False, ([1, 2, 3, 1],), dict(), False),
        (True, False, ([2**30],), dict(), True),
        # (True, False, ([2**31],), dict(), True),
        # (True, False, ([2**33],), dict(), True),
        # (True, False, ([2**34],), dict(), True),
        # (True, False, ([2**35],), dict(), True),
      )
    ),
    ((check_permutation, check_permutation_better, check_permutation_sort,),
      (
        (True, False, ('',''), dict(), True),
        (True, False, ('a','a'), dict(), True),
        (True, False, ('aa','aa'), dict(), True),
        (True, False, ('ab','ab'), dict(), True),
        (True, False, ('ab','ba'), dict(), True),
        (True, False, ('abc','bca'), dict(), True),
        (True, False, ('aabc','baca'), dict(), True),
        (True, False, ('','a'), dict(), False),
        (True, False, ('a','aa'), dict(), False),
        (True, False, ('ab','aba'), dict(), False),
      )
    ),
    ((urlify, urlify_manual_better, urlify_manual, urlify_manual_immutable, ),
      (
        (True, False, ('',), dict(), ''),
        (True, False, ('a',), dict(), 'a'),
        (True, False, (' ',), dict(), '%20'),
        (True, False, ('  ',), dict(), '%20%20'),
        (True, False, (' a',), dict(), '%20a'),
        (True, False, ('a ',), dict(), 'a%20'),
        (True, False, (' a ',), dict(), '%20a%20'),
        (True, False, ('Mr John Smith',), dict(), 'Mr%20John%20Smith'),
      )
    ),
    ((palindrome_permutation,),
      (
        (True, False, ('',), dict(), True),
        (True, False, ('a',), dict(), True),
        (True, False, ('aa',), dict(), True),
        (True, False, ('aaa',), dict(), True),
        (True, False, ('aba',), dict(), True),
        (True, False, ('aab',), dict(), True),
        (True, False, ('abba',), dict(), True),
        (True, False, ('aabb',), dict(), True),
        (True, False, ('abbcccc',), dict(), True),
        (True, False, ('ab',), dict(), False),
        (True, False, ('abbb',), dict(), False),
        (True, False, ('abc',), dict(), False),
        (True, False, ('Tact Coa',), dict(), True),
      )
    ),
    ((one_away, one_away_better,),
      (
        (True, False, ('',''), dict(), True),
        (True, False, ('a','a'), dict(), True),
        (True, False, ('ab','ab'), dict(), True),
        (True, False, ('ab','a'), dict(), True),
        (True, False, ('','a'), dict(), True),
        (True, False, ('a','ab'), dict(), True),
        (True, False, ('a','aaa'), dict(), False),
        (True, False, ('','ab'), dict(), False),
        (True, False, ('a','abc'), dict(), False),
        (True, False, ('abx','abb'), dict(), True),
        (True, False, ('abx','abc'), dict(), True),
        (True, False, ('axx','abb'), dict(), False),
        (True, False, ('pale','ple'), dict(), True),
        (True, False, ('pales','pale'), dict(), True),
        (True, False, ('pale','bale'), dict(), True),
        (True, False, ('pale','bake'), dict(), False),
      )
    ),
    ((string_compression,),
      (
        (True, False, ('',), dict(), ''),
        (True, False, ('a',), dict(), 'a'),
        (True, False, ('aa',), dict(), 'aa'),
        (True, False, ('ab',), dict(), 'ab'),
        (True, False, ('aaa',), dict(), 'a3'),
        (True, False, ('aaab',), dict(), 'aaab'),
        (True, False, ('aaaab',), dict(), 'a4b1'),
        (True, False, ('aabbcc',), dict(), 'aabbcc'),
        (True, False, ('aaabbcc',), dict(), 'a3b2c2'),
        (True, False, ('aabcccccaaa',), dict(), 'a2b1c5a3'),
      )
    ),
    ((rotate_matrix,rotate_matrix_in_place,),
      (
        (True, False, ([[]],), dict(), [[]]),
        (True, False, ([[0]],), dict(), [[0]]),
        (True, False, ([[0,0],[0,0]],), dict(), [[0,0],[0,0]]),
        (True, False, ([[1,2],[3,4]],), dict(), [[3,1],[4,2]]),
        (True, False, ([[1,2,3],[4,5,6],[7,8,9]],), dict(), [[7,4,1],[8,5,2],[9,6,3]]),
        (True, False, ([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],), dict(), [[13,9,5,1],[14,10,6,2],[15,11,7,3],[16,12,8,4]]),
        (True, False, ([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]],), dict(), [[21,16,11,6,1],[22,17,12,7,2],[23,18,13,8,3],[24,19,14,9,4],[25,20,15,10,5]]),
      )
    ),
    ((zero_matrix,zero_matrix_better,zero_matrix_in_place,),
      (
        (True, False, ([[]],), dict(), [[]]),
        (True, False, ([[0]],), dict(), [[0]]),
        (True, False, ([[1,2],[3,4]],), dict(), [[1,2],[3,4]]),
        (True, False, ([[0,2],[3,4]],), dict(), [[0,0],[0,4]]),
        (True, False, ([[1,0],[3,4]],), dict(), [[0,0],[3,0]]),
        (True, False, ([[0,0],[3,4]],), dict(), [[0,0],[0,0]]),
        (True, False, ([[1,2,3],[4,5,6],[7,8,9]],), dict(), [[1,2,3],[4,5,6],[7,8,9]]),
        (True, False, ([[0,2,3],[4,5,6],[7,8,9]],), dict(), [[0,0,0],[0,5,6],[0,8,9]]),
        (True, False, ([[1,0,3],[4,5,6],[7,8,9]],), dict(), [[0,0,0],[4,0,6],[7,0,9]]),
        (True, False, ([[1,2,3],[4,0,6],[7,8,9]],), dict(), [[1,0,3],[0,0,0],[7,0,9]]),
        (True, False, ([[0,2,3],[4,5,6],[7,8,0]],), dict(), [[0,0,0],[0,5,0],[0,0,0]]),
      )
    ),
    ((is_substring,),
      (
        (True, False, ('',''), dict(), True),
        (True, False, ('a',''), dict(), True),
        (True, False, ('ab',''), dict(), True),
        (True, False, ('','a'), dict(), False),
        (True, False, ('a','a'), dict(), True),
        (True, False, ('ab','a'), dict(), True),
        (True, False, ('ba','a'), dict(), True),
        (True, False, ('ab','ab'), dict(), True),
        (True, False, ('ab','ba'), dict(), False),
        (True, False, ('ba','ab'), dict(), False),
        (True, False, ('abb','bb'), dict(), True),
        (True, False, ('bab','bb'), dict(), False),
        (True, False, ('bba','bb'), dict(), True),
        (True, False, ('erbottlewaterbottlewat','waterbottle'), dict(), True),
      )
    ),
    ((string_rotation,),
      (
        (True, False, ('',''), dict(), True),
        (True, False, ('a',''), dict(), False),
        (True, False, ('ab',''), dict(), False),
        (True, False, ('','a'), dict(), False),
        (True, False, ('a','a'), dict(), True),
        (True, False, ('ab','a'), dict(), False),
        (True, False, ('ba','a'), dict(), False),
        (True, False, ('ab','ab'), dict(), True),
        (True, False, ('ab','ba'), dict(), True),
        (True, False, ('ba','ab'), dict(), True),
        (True, False, ('abb','bb'), dict(), False),
        (True, False, ('bab','bb'), dict(), False),
        (True, False, ('bba','bb'), dict(), False),
        (True, False, ('waterbottle','erbottlewat'), dict(), True),
      )
    ),
  )
  
  ################################################
  # Ch2
  ################################################
  # Test for 2.3. delete_middle_node_object requires an actual object to be passed while existing already.
  _2_3_singly_linked_list_ab = SinglyLinkedList('ab')
  _2_3_singly_linked_list_ab_node_b = _2_3_singly_linked_list_ab.first_node.next_node
  
  _2_3_singly_linked_list_abc = SinglyLinkedList('abc')
  _2_3_singly_linked_list_abc_node_b = _2_3_singly_linked_list_abc.first_node.next_node
  
  _2_3_singly_linked_list_abcd_1 = SinglyLinkedList('abcd')
  _2_3_singly_linked_list_abcd_1_node_b = _2_3_singly_linked_list_abcd_1.first_node.next_node
  
  _2_3_singly_linked_list_abcd_2 = SinglyLinkedList('abcd')
  _2_3_singly_linked_list_abcd_2_node_c = _2_3_singly_linked_list_abcd_2.first_node.next_node.next_node
  
  #  Test for 2.7. intersection_ll requires the same Node to exist in different LinkedLists
  _2_7_singly_linked_list_a_01 = SinglyLinkedList([1,2,3,4])
  _2_7_singly_linked_list_b_01 = SinglyLinkedList([1,2,9])
  _2_7_singly_linked_list_b_01.first_node.next_node.next_node = _2_7_singly_linked_list_a_01.first_node.next_node.next_node  # Potentially causes previous line to memory leak the Node(9) object.
  
  _2_7_singly_linked_list_a_02 = SinglyLinkedList([1,2,3,4])
  _2_7_singly_linked_list_b_02 = SinglyLinkedList([1])
  _2_7_singly_linked_list_b_02.first_node = _2_7_singly_linked_list_a_02.first_node
  
  #  Test for 2.8. ll_loop_detection requires the same Node to be linked to in the LinkedList more than once
  _2_8_singly_linked_list_01 = SinglyLinkedList([1,2,3,4])
  _2_8_singly_linked_list_01.first_node.next_node.next_node.next_node = _2_8_singly_linked_list_01.first_node
  
  _2_8_singly_linked_list_02 = SinglyLinkedList([1])
  _2_8_singly_linked_list_02.first_node.next_node = _2_8_singly_linked_list_02.first_node
  
  _2_8_singly_linked_list_03 = SinglyLinkedList('abcde')
  _2_8_singly_linked_list_03.first_node.next_node.next_node.next_node.next_node.next_node = _2_8_singly_linked_list_03.first_node.next_node.next_node
  
  checks_ch2 = (  
    ((remove_dups, remove_dups_no_buffer,),
      (
        (True, False, (SinglyLinkedList(),), dict(), str(SinglyLinkedList())),
        (True, False, (SinglyLinkedList([1]),), dict(), str(SinglyLinkedList([1]))),
        (True, False, (SinglyLinkedList([1,2]),), dict(), str(SinglyLinkedList([1,2]))),
        (True, False, (SinglyLinkedList([1,1]),), dict(), str(SinglyLinkedList([1]))),
        (True, False, (SinglyLinkedList([1,2,1]),), dict(), str(SinglyLinkedList([1,2]))),
        (True, False, (SinglyLinkedList([1,1,1,1,1,1,1,1,1]),), dict(), str(SinglyLinkedList([1]))),
        (True, False, (SinglyLinkedList([1,1,1,1,1,1,1,1,1,2]),), dict(), str(SinglyLinkedList([1,2]))),
        (True, False, (SinglyLinkedList([1,2,1,1,1,1,1,1,1,1]),), dict(), str(SinglyLinkedList([1,2]))),
        (True, False, (SinglyLinkedList([1,1,1,2,2,2,3,3,3,4,4,4]),), dict(), str(SinglyLinkedList([1,2,3,4]))),
        (True, False, (SinglyLinkedList([1,2,3,4,1,2,3,4,1,2,3,4]),), dict(), str(SinglyLinkedList([1,2,3,4]))),
      )
    ),
    ((return_kth_to_last, return_kth_to_last_window, return_kth_to_last_cheat),
      (
        (True, False, (SinglyLinkedList('a'), 1), dict(), 'a'),
        (True, False, (SinglyLinkedList('ab'), 1), dict(), 'b'),
        (True, False, (SinglyLinkedList('ab'), 2), dict(), 'a'),
        (True, False, (SinglyLinkedList('abc'), 1), dict(), 'c'),
        (True, False, (SinglyLinkedList('abc'), 2), dict(), 'b'),
        (True, False, (SinglyLinkedList('abc'), 3), dict(), 'a'),
        (True, False, (SinglyLinkedList('aaa'), 1), dict(), 'a'),
        (True, False, (SinglyLinkedList('aaa'), 2), dict(), 'a'),
        (True, False, (SinglyLinkedList('aaa'), 3), dict(), 'a'),
      )
    ),
    ((delete_middle_node,),
      (
        (True, False, (SinglyLinkedList('ab'), 'b'), dict(), str(SinglyLinkedList('a'))),
        (True, False, (SinglyLinkedList('abc'), 'b'), dict(), str(SinglyLinkedList('ac'))),
        (True, False, (SinglyLinkedList('abbc'), 'b'), dict(), str(SinglyLinkedList('ac'))),
        (True, False, (SinglyLinkedList('abbbc'), 'b'), dict(), str(SinglyLinkedList('ac'))),
        (True, False, (SinglyLinkedList('abcb'), 'b'), dict(), str(SinglyLinkedList('ac'))),
        (True, False, (SinglyLinkedList('abcbb'), 'b'), dict(), str(SinglyLinkedList('ac'))),
        (True, False, (SinglyLinkedList('abcbbb'), 'b'), dict(), str(SinglyLinkedList('ac'))),
        (True, False, (SinglyLinkedList('abbbcbbb'), 'b'), dict(), str(SinglyLinkedList('ac'))),
      )
    ),
    ((delete_middle_node_object,),
      (
        (False, False, (_2_3_singly_linked_list_ab, _2_3_singly_linked_list_ab_node_b), dict(), str(SinglyLinkedList('a'))),
        (False, False, (_2_3_singly_linked_list_abc, _2_3_singly_linked_list_abc_node_b), dict(), str(SinglyLinkedList('ac'))),
        (False, False, (_2_3_singly_linked_list_abcd_1, _2_3_singly_linked_list_abcd_1_node_b), dict(), str(SinglyLinkedList('acd'))),
        (False, False, (_2_3_singly_linked_list_abcd_2, _2_3_singly_linked_list_abcd_2_node_c), dict(), str(SinglyLinkedList('abd'))),
      )
    ),
    ((partition_ll,),
      (
        (True, False, (SinglyLinkedList([]), 0), dict(), str(SinglyLinkedList([]))),
        (True, False, (SinglyLinkedList([1]), 0), dict(), str(SinglyLinkedList([1]))),
        (True, False, (SinglyLinkedList([1]), 1), dict(), str(SinglyLinkedList([1]))),
        (True, False, (SinglyLinkedList([1]), 2), dict(), str(SinglyLinkedList([1]))),
        (True, False, (SinglyLinkedList([1,2]), 0), dict(), str(SinglyLinkedList([1,2]))),
        (True, False, (SinglyLinkedList([1,2]), 1), dict(), str(SinglyLinkedList([1,2]))),
        (True, False, (SinglyLinkedList([1,2]), 2), dict(), str(SinglyLinkedList([1,2]))),
        (True, False, (SinglyLinkedList([1,2]), 3), dict(), str(SinglyLinkedList([1,2]))),
        (True, False, (SinglyLinkedList([2,1]), 0), dict(), str(SinglyLinkedList([2,1]))),
        (True, False, (SinglyLinkedList([2,1]), 1), dict(), str(SinglyLinkedList([2,1]))),
        (True, False, (SinglyLinkedList([2,1]), 2), dict(), str(SinglyLinkedList([1,2]))),
        (True, False, (SinglyLinkedList([2,1]), 3), dict(), str(SinglyLinkedList([2,1]))),
        (True, False, (SinglyLinkedList([1,2,3]), 0), dict(), str(SinglyLinkedList([1,2,3]))),
        (True, False, (SinglyLinkedList([1,2,3]), 1), dict(), str(SinglyLinkedList([1,2,3]))),
        (True, False, (SinglyLinkedList([1,2,3]), 2), dict(), str(SinglyLinkedList([1,2,3]))),
        (True, False, (SinglyLinkedList([1,2,3]), 3), dict(), str(SinglyLinkedList([1,2,3]))),
        (True, False, (SinglyLinkedList([3,2,1]), 0), dict(), str(SinglyLinkedList([3,2,1]))),
        (True, False, (SinglyLinkedList([3,2,1]), 1), dict(), str(SinglyLinkedList([3,2,1]))),
        (True, False, (SinglyLinkedList([3,2,1]), 2), dict(), str(SinglyLinkedList([1,3,2]))),
        (True, False, (SinglyLinkedList([3,2,1]), 3), dict(), str(SinglyLinkedList([2,1,3]))),
        (True, False, (SinglyLinkedList([3,2,1]), 4), dict(), str(SinglyLinkedList([3,2,1]))),
        (True, False, (SinglyLinkedList([1,3,1,2,1]), 2), dict(), str(SinglyLinkedList([1,1,1,3,2]))),
        (True, False, (SinglyLinkedList([3,5,8,5,10,2,1]), 5), dict(), str(SinglyLinkedList([3,2,1,5,8,5,10]))),
      )
    ),
    ((sum_lists,),
      (
        (True, False, (SinglyLinkedList([]), SinglyLinkedList([])), dict(), str(SinglyLinkedList([]))),
        (True, False, (SinglyLinkedList([1]), SinglyLinkedList([])), dict(), str(SinglyLinkedList([1]))),
        (True, False, (SinglyLinkedList([]), SinglyLinkedList([1])), dict(), str(SinglyLinkedList([1]))),
        (True, False, (SinglyLinkedList([1,2]), SinglyLinkedList([])), dict(), str(SinglyLinkedList([1,2]))),
        (True, False, (SinglyLinkedList([]), SinglyLinkedList([1,2])), dict(), str(SinglyLinkedList([1,2]))),
        (True, False, (SinglyLinkedList([1]), SinglyLinkedList([1])), dict(), str(SinglyLinkedList([2]))),
        (True, False, (SinglyLinkedList([1,9]), SinglyLinkedList([1])), dict(), str(SinglyLinkedList([2,9]))),
        (True, False, (SinglyLinkedList([1]), SinglyLinkedList([1,9])), dict(), str(SinglyLinkedList([2,9]))),
        (True, False, (SinglyLinkedList([1,7,8,9]), SinglyLinkedList([1])), dict(), str(SinglyLinkedList([2,7,8,9]))),
        (True, False, (SinglyLinkedList([1]), SinglyLinkedList([1,7,8,9])), dict(), str(SinglyLinkedList([2,7,8,9]))),
        (True, False, (SinglyLinkedList([1]), SinglyLinkedList([9])), dict(), str(SinglyLinkedList([0,1]))),
        (True, False, (SinglyLinkedList([9]), SinglyLinkedList([1])), dict(), str(SinglyLinkedList([0,1]))),
        (True, False, (SinglyLinkedList([3]), SinglyLinkedList([9])), dict(), str(SinglyLinkedList([2,1]))),
        (True, False, (SinglyLinkedList([9]), SinglyLinkedList([3])), dict(), str(SinglyLinkedList([2,1]))),
        (True, False, (SinglyLinkedList([1]), SinglyLinkedList([9,9,9,9])), dict(), str(SinglyLinkedList([0,0,0,0,1]))),
        (True, False, (SinglyLinkedList([9,9,9,9]), SinglyLinkedList([1])), dict(), str(SinglyLinkedList([0,0,0,0,1]))),
        (True, False, (SinglyLinkedList([9]), SinglyLinkedList([9,9,9,9])), dict(), str(SinglyLinkedList([8,0,0,0,1]))),
        (True, False, (SinglyLinkedList([9,9,9,9]), SinglyLinkedList([9])), dict(), str(SinglyLinkedList([8,0,0,0,1]))),
        (True, False, (SinglyLinkedList([2,2,3,4]), SinglyLinkedList([9,9,9,9])), dict(), str(SinglyLinkedList([1,2,3,4,1]))),
        (True, False, (SinglyLinkedList([9,9,9,9]), SinglyLinkedList([2,2,3,4])), dict(), str(SinglyLinkedList([1,2,3,4,1]))),
        (True, False, (SinglyLinkedList([7,1,6]), SinglyLinkedList([5,9,2])), dict(), str(SinglyLinkedList([2,1,9]))),
      )
    ),
    ((sum_lists_reverse_representation,),
      (
        (True, False, (SinglyLinkedList([]), SinglyLinkedList([])), dict(), str(SinglyLinkedList([]))),
        (True, False, (SinglyLinkedList([1]), SinglyLinkedList([])), dict(), str(SinglyLinkedList([1]))),
        (True, False, (SinglyLinkedList([]), SinglyLinkedList([1])), dict(), str(SinglyLinkedList([1]))),
        (True, False, (SinglyLinkedList([1,2]), SinglyLinkedList([])), dict(), str(SinglyLinkedList([1,2]))),
        (True, False, (SinglyLinkedList([]), SinglyLinkedList([1,2])), dict(), str(SinglyLinkedList([1,2]))),
        (True, False, (SinglyLinkedList([1]), SinglyLinkedList([1])), dict(), str(SinglyLinkedList([2]))),
        (True, False, (SinglyLinkedList([1,8]), SinglyLinkedList([1])), dict(), str(SinglyLinkedList([1,9]))),
        (True, False, (SinglyLinkedList([1]), SinglyLinkedList([1,8])), dict(), str(SinglyLinkedList([1,9]))),
        (True, False, (SinglyLinkedList([1,7,8,8]), SinglyLinkedList([1])), dict(), str(SinglyLinkedList([1,7,8,9]))),
        (True, False, (SinglyLinkedList([1]), SinglyLinkedList([1,7,8,8])), dict(), str(SinglyLinkedList([1,7,8,9]))),
        (True, False, (SinglyLinkedList([1]), SinglyLinkedList([9])), dict(), str(SinglyLinkedList([1,0]))),
        (True, False, (SinglyLinkedList([9]), SinglyLinkedList([1])), dict(), str(SinglyLinkedList([1,0]))),
        (True, False, (SinglyLinkedList([3]), SinglyLinkedList([9])), dict(), str(SinglyLinkedList([1,2]))),
        (True, False, (SinglyLinkedList([9]), SinglyLinkedList([3])), dict(), str(SinglyLinkedList([1,2]))),
        (True, False, (SinglyLinkedList([1]), SinglyLinkedList([9,9,9,9])), dict(), str(SinglyLinkedList([1,0,0,0,0]))),
        (True, False, (SinglyLinkedList([9,9,9,9]), SinglyLinkedList([1])), dict(), str(SinglyLinkedList([1,0,0,0,0]))),
        (True, False, (SinglyLinkedList([9]), SinglyLinkedList([9,9,9,9])), dict(), str(SinglyLinkedList([1,0,0,0,8]))),
        (True, False, (SinglyLinkedList([9,9,9,9]), SinglyLinkedList([9])), dict(), str(SinglyLinkedList([1,0,0,0,8]))),
        (True, False, (SinglyLinkedList([4,3,2,2]), SinglyLinkedList([9,9,9,9])), dict(), str(SinglyLinkedList([1,4,3,2,1]))),
        (True, False, (SinglyLinkedList([9,9,9,9]), SinglyLinkedList([4,3,2,2])), dict(), str(SinglyLinkedList([1,4,3,2,1]))),
        (True, False, (SinglyLinkedList([6,1,7]), SinglyLinkedList([2,9,5])), dict(), str(SinglyLinkedList([9,1,2]))),
      )
    ),
    ((palindrome_ll, palindrome_ll_manual,),
      (
        (True, False, (SinglyLinkedList([]),), dict(), True),
        (True, False, (SinglyLinkedList([1]),), dict(), True),
        (True, False, (SinglyLinkedList([1,1]),), dict(), True),
        (True, False, (SinglyLinkedList([1,1,1]),), dict(), True),
        (True, False, (SinglyLinkedList([1,1,1,1]),), dict(), True),
        (True, False, (SinglyLinkedList([1,2]),), dict(), False),
        (True, False, (SinglyLinkedList([1,2,1]),), dict(), True),
        (True, False, (SinglyLinkedList([1,2,1,1]),), dict(), False),
        (True, False, (SinglyLinkedList([1,2,2,1]),), dict(), True),
        (True, False, (SinglyLinkedList([1,2,3,2,1]),), dict(), True),
        (True, False, (SinglyLinkedList([1,2,3,2,2,1]),), dict(), False),
        (True, False, (SinglyLinkedList([1,2,3,3,2,1]),), dict(), True),
      )
    ),
    ((intersection_ll, intersection_ll_manual, intersection_ll_manual_generator,),
      (
        (False, True, (SinglyLinkedList([]), SinglyLinkedList([])), dict(), None),
        (False, True, (SinglyLinkedList([1]), SinglyLinkedList([1])), dict(), None),
        (False, True, (SinglyLinkedList([1,2]), SinglyLinkedList([1,2])), dict(), None),
        (False, True, (_2_7_singly_linked_list_a_01, _2_7_singly_linked_list_b_01), dict(), _2_7_singly_linked_list_a_01.first_node.next_node.next_node),
        (False, True, (_2_7_singly_linked_list_a_02, _2_7_singly_linked_list_b_02), dict(), _2_7_singly_linked_list_a_02.first_node),
      )
    ),
    ((ll_loop_detection, ll_loop_detection_manual,),
      (
        (False, True, (SinglyLinkedList([]),), dict(), None),
        (False, True, (SinglyLinkedList([1]),), dict(), None),
        (False, True, (SinglyLinkedList([1,2]),), dict(), None),
        (False, True, (_2_8_singly_linked_list_01,), dict(), _2_8_singly_linked_list_01.first_node),
        (False, True, (_2_8_singly_linked_list_02,), dict(), _2_8_singly_linked_list_02.first_node),
        (False, True, (_2_8_singly_linked_list_03,), dict(), _2_8_singly_linked_list_03.first_node.next_node.next_node),
      )
    ),
  )
  
  ################################################
  # Ch3
  ################################################
  _3_1_bounding_indices_01 = [0,2,5,9]  # Last valid index for each stack
  _3_1_arr_01 = [None] * (_3_1_bounding_indices_01[-1]+1)
  
  checks_ch3 = (  
    ((three_in_one,),
      (
        # For these tests, we disable duplication_flag, and have the tests build on top of each other, since `arr` is mutable.
        (False, False, (None, None, -1), dict(), ValueError()),  # arr is None
        (False, False, (_3_1_arr_01, None, -1), dict(), ValueError()),  # bounding_indices is None
        (False, False, (_3_1_arr_01, [-1]*(len(_3_1_arr_01)), -1), dict(), ValueError()),  # len(arr) != bounding_indices[-1]+1:
        (False, False, (_3_1_arr_01, [len(_3_1_arr_01)+1]*(len(_3_1_arr_01)), -1), dict(), ValueError()),  # len(arr) != bounding_indices[-1]+1:
        (False, False, (_3_1_arr_01, list(reversed(_3_1_bounding_indices_01)), -1), dict(), ValueError()),  # bounding_indices not asc
        (False, False, (_3_1_arr_01, _3_1_bounding_indices_01, -1), dict(), ValueError()),  # request < 0 or request > 3
        (False, False, (_3_1_arr_01, _3_1_bounding_indices_01, 4), dict(), ValueError()),  # request < 0 or request > 3
        (False, False, (_3_1_arr_01, _3_1_bounding_indices_01, 1), {'operate_on': 0, 'push_data': '_3_1_arr_01_stack_00_data_00'}, None),  # push
        (False, False, (_3_1_arr_01, _3_1_bounding_indices_01, 1), {'operate_on': 0, 'push_data': '_3_1_arr_01_stack_00_data_00'}, StackIsFullError()),  # StackIsFullError
        (False, False, (_3_1_arr_01, _3_1_bounding_indices_01, 3), {'operate_on': 0}, '_3_1_arr_01_stack_00_data_00'),  # peek
        (False, False, (_3_1_arr_01, _3_1_bounding_indices_01, 2), {'operate_on': 0}, '_3_1_arr_01_stack_00_data_00'),  # pop
        (False, False, (_3_1_arr_01, _3_1_bounding_indices_01, 3), {'operate_on': 0}, None),  # peek
        (False, False, (_3_1_arr_01, _3_1_bounding_indices_01, 3), {'operate_on': 1}, None),  # peek
        (False, False, (_3_1_arr_01, _3_1_bounding_indices_01, 1), {'operate_on': 2, 'push_data': '_3_1_arr_01_stack_02_data_00'}, None),  # push
        (False, False, (_3_1_arr_01, _3_1_bounding_indices_01, 1), {'operate_on': 2, 'push_data': '_3_1_arr_01_stack_02_data_01'}, None),  # push
        (False, False, (_3_1_arr_01, _3_1_bounding_indices_01, 1), {'operate_on': 2, 'push_data': '_3_1_arr_01_stack_02_data_02'}, None),  # push
        (False, False, (_3_1_arr_01, _3_1_bounding_indices_01, 3), {'operate_on': 2}, '_3_1_arr_01_stack_02_data_02'),  # peek
        (False, False, (_3_1_arr_01, _3_1_bounding_indices_01, 2), {'operate_on': 2}, '_3_1_arr_01_stack_02_data_02'),  # pop
        (False, False, (_3_1_arr_01, _3_1_bounding_indices_01, 2), {'operate_on': 2}, '_3_1_arr_01_stack_02_data_01'),  # pop
        (False, False, (_3_1_arr_01, _3_1_bounding_indices_01, 2), {'operate_on': 2}, '_3_1_arr_01_stack_02_data_00'),  # pop
        (False, False, (_3_1_arr_01, _3_1_bounding_indices_01, 2), {'operate_on': 2}, None),  # pop
      )
    ),
  )
  
  ################################################
  # Combine chapters
  ################################################
  checks_tuples = (
    checks_ch1,
    # checks_ch2,
    # checks_ch3,
  )
  for checks_tuple_ in checks_tuples:
    assert isinstance(checks_tuple_, tuple)
  
  checks = tuple(check_instance for tuple_ in checks_tuples for check_instance in tuple_)  # Merge all chapter checks tuples. They are separate to enable easier collapsing in the editor.

  ################################################
  # Run tests and output results
  ################################################
  # Log output flags 
  #   -- TODO: change to actual flags/enumeration system.
  #   -- TODO: Add optional commandline parameterization.
  PRINT_FUNCTIONS = True
  PRINT_FUNCTION_ANNOTATIONS = False
  #PRINT_TESTS = True  # TODO DEBUG: Disable
  PRINT_TESTS = False
  PRINT_FAILS = True

  # Init
  fails = dict()
  func_count = test_count = 0
  func_fail_count = test_fail_count = 0

  # Loop through each check
  for check_num, (funcs, tests) in enumerate(checks, 1):
    assert isinstance(funcs, tuple)
    assert isinstance(tests, tuple)
    
    # Loop through each function in current check
    for func_num, func_ in enumerate(funcs, 1):
      assert isinstance(func_, types.FunctionType)
      # Note: certain builtin functions aren't types.FunctionType. See: https://stackoverflow.com/a/624948/7121931
      #
      # >>> import types
      # >>> types.FunctionType
      # <class 'function'>
      #
      # >>> def f(): pass
      #
      # >>> isinstance(f, types.FunctionType)
      # True
      #
      # >>> isinstance(lambda x : None, types.FunctionType)
      # True
      #
      # >>> type(zip), isinstance(zip, types.FunctionType)
      # (<class 'type'>, False)
      #
      # >>> type(open), isinstance(open, types.FunctionType)
      # (<class 'builtin_function_or_method'>, False)
      #
      # >>> type(random.shuffle), isinstance(random.shuffle, types.FunctionType)
      # (<class 'method'>, False)
      
      func_count += 1

      # Function output
      if PRINT_FUNCTIONS or PRINT_TESTS or PRINT_FAILS:
        top_level_log_output = f'Check #{check_num:02}.{func_num:02}: {func_.__name__}{" "+str(func_.__annotations__) if PRINT_FUNCTION_ANNOTATIONS else ""}:'
        if PRINT_FUNCTIONS or PRINT_TESTS:
          print(top_level_log_output[:None if PRINT_TESTS else -1])  # Do not print trailing ':' if not PRINT_TESTS

      # Loop through each test case in current check
      #   -- note: a check can have multiple functions, but all of these functions will receive the same tests
      for test_case_num, (duplication_flag, require_is_match, test_input_args, test_input_kwargs, expected_output) in enumerate(tests, 1):
        # Ensure all the test case variables are of the correct type
        assert isinstance(duplication_flag, bool)
        assert isinstance(require_is_match, bool)
        assert isinstance(test_input_args, tuple)
        assert isinstance(test_input_kwargs, dict)
      
        test_count += 1
        
        # Copy test case inputs, if `duplication_flag` is True
        if duplication_flag:
          copy_of_test_input_args = tuple(copy.copy(test_input__) for test_input__ in test_input_args)  # TODO: Technically this should be a deep_copy, depending on the function we are testing.
          copy_of_test_input_kwargs = {key: copy.copy(test_input__) for key, test_input__ in test_input_kwargs.items()}  # TODO: Technically this should be a deep_copy, depending on the function we are testing.
      
        #actual_output = func_(*test_input_args)  # Reusing the same test inputs is bad when they contain mutable elements that solutions can modify as part of arriving at the expected output.
        # Get the output for the function being tested.
        try:
          func_raised_exception = False
          if duplication_flag:
            actual_output = func_(*copy_of_test_input_args, **copy_of_test_input_kwargs)
          else:
            actual_output = func_(*test_input_args, **test_input_kwargs)
        except:
          func_raised_exception = True
          curr_exc_info = sys.exc_info()
          if type(expected_output) == curr_exc_info[0]:  # If we expected an exception to be raised as part of this testcase.
            actual_output = curr_exc_info[1]
          else:  # We did not expect an exception, so re-raise it.
            raise

        # Test if the output matches the expected output. Depending on whether `require_is_match` is set or not, we do this using `is` or `==`.
        if not func_raised_exception:
          if require_is_match:
            test_succeeded = actual_output is expected_output
          else:
            test_succeeded = actual_output == expected_output
        else:
          test_succeeded = True

        # Test case output
        if PRINT_TESTS or (PRINT_FAILS and not test_succeeded):
          log_output = \
            f'    Test case #{test_case_num:02}: {"PASSED" if test_succeeded else "FAILED <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"}\n' \
            f'      Input args:       {test_input_args}\n' \
            f'      Input kwargs:     {test_input_kwargs}\n' \
            f'      Input args str:   {tuple(str(input_) for input_ in test_input_args)}\n' \
            f'      Input kwargs str: {", ".join(str(key) + ": " + str(input_) for key,input_ in test_input_kwargs.items())}\n' \
            f'      Expected output:  {repr(expected_output)}\n' \
            f'      Actual output:    {repr(actual_output)}\n'
          if PRINT_TESTS:
            print(log_output)

        # Tracking failed test cases 
        if not test_succeeded:
          # If this function already had a failed test case (in this `checks` iteration)
          if (check_num, func_num) in fails:
            test_fail_count += 1
            if PRINT_FAILS:
              fails[(check_num, func_num)][1].append(log_output)
              
          # If this was the first failed test case for this function (in this `checks` iteration)
          else:
            func_fail_count += 1
            test_fail_count += 1
            if PRINT_FAILS:
              fails[(check_num, func_num)] = (top_level_log_output, [log_output])
            else:
              fails[(check_num, func_num)] = None

    # Separator between function outputs
    if PRINT_FUNCTIONS or PRINT_TESTS:
      print('--')

  # Final output
  if fails:
    if PRINT_FAILS:
      print('# Fails:')
      print('--\n'+'--\n'.join(fail_[0]+'\n'+'\n'.join(fail_[1]) for fail_ in fails.values()))
    print(f'\n{func_fail_count}/{func_count} functions failed, with {test_fail_count}/{test_count} failed tests.')
  else:
    print(f'\nAll tests succeeded. ({func_count} functions, with {test_count} tests.)')

  #sys.stdout.close()  # Close stdout, if we rerouted it to a file earlier

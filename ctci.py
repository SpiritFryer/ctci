#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Python3 solutions and test cases for Cracking the Coding Interview, 6th Edition

Python3.9.5 was used. -- SpiritFryer
"""

__author__ = 'SpiritFryer'
__credits__ = ['Cracking the Coding Interview, 6th Edition']

# ctci.py
# Cracking the Coding Interview, 6th Edition
# Started: 2021-11-09
# https://github.com/SpiritFryer
# 
# Also see:
# https://github.com/careercup/CtCI-6th-Edition-Python/

################################################
# Imports and general-use functions
################################################
import itertools
from typing import Any, List


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


################################################
# Ch1
################################################
def is_unique(s: str) -> bool:
  """1. Is Unique: Implement an algorithm to determine if a string has all unique characters.

  What if you cannot use additional data structures?
  """
  # set(s): O(n)
  # len(s): O(n), but we can track it while building set(s)
  # len(set(s)): O(n), but we can track it while building set(s)
  #
  # Time: O(n)
  # Space: O(n), but likely much less
  return len(set(s)) == len(s)

def is_unique_no_data_structures(s: str) -> bool:
  """1. Is Unique: Implement an algorithm to determine if a string has all unique characters.

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
  """2. Check Permutation: Given two strings, write a method to decide if one is a permutation of the other."""
  # TODO: complexity analysis
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


def urlify(s: str) -> str:
  """3. URLify: Write a method to replace all spaces in a string with '%20.

  You may assume that the string has sufficient space at the end to hold the additional characters,
  and that you are given the "true" length of the string.
  (Note: If implementing in Java, please use a character array so that you can perform this operation in place.)

  EXAMPLE
    Input: "Mr John Smith "J 13
    Output: "Mr%20J ohn%20Smith"
  """
  # TODO: complexity analysis
  return s.replace(' ', '%20')

def urlify_manual(s: str) -> str:
  """3. URLify: Write a method to replace all spaces in a string with '%20.

  You may assume that the string has sufficient space at the end to hold the additional characters,
  and that you are given the "true" length of the string.
  (Note: If implementing in Java, please use a character array so that you can perform this operation in place.)

  EXAMPLE
    Input: "Mr John Smith "J 13
    Output: "Mr%20J ohn%20Smith"
  """
  # TODO: complexity analysis
  output = ''
  for char_ in s:
    if char_ == ' ':
      output += '%20'
    else:
      output += char_
  return output


def palindrome_permutation(s: str) -> bool:
  """4. Palindrome Permutation: Given a string, write a function to check if it is a permutation of a palindrome.

  A palindrome is a word or phrase that is the same forwards and backwards. A permutation is a rearrangement of letters.
  The palindrome does not need to be limited to just dictionary words.

  EXAMPLE
    Input: Tact Coa
    Output: True (permutations: "taco cat". "atco cta". etc.)
  """
  # TODO: complexity analysis

  # At most 1 char could show up an odd number of times. All other chars must show up an even number of times.
  parity = dict()  # parity[char_] == True: even number of times; parity[char_] == False: odd number of times.
  count_odd = 0

  s_condensed = ''.join(s.lower().split())  # Remove all whitespace, ignore case

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
  """5. One Away: Given two strings, write a function to check if they are one edit (or zero edits) away.

  There are three types of edits that can be performed on strings:
    insert a character,
    remove a character,
    or replace a character.

  EXAMPLE
    pale, ple -> true
    pales. pale -> true
    pale. bale -> true
    pale. bake -> false
  """
  # TODO: complexity analysis

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


def string_compression(s: str) -> str:
  """6. String Compression: Perform basic string compression using the counts of repeated characters.

  For example, the string aabcccccaaa would become a2b1c5a3.
  If the "compressed" string would not become smaller than the original string, return the original string.

  You can assume the string has only uppercase and lowercase letters (a - z).
  """
  # TODO: complexity analysis

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
  """7. Rotate Matrix: Rotate an NxN matrix by 90-degrees clockwise.

  Given an image represented by an NxN matrix, where each pixel in the image is 4 bytes, write a method to rotate the
  image by 90 degrees. Can you do this in place?
  """
  # TODO: complexity analysis
  if len(m) <= 1:
    return m

  new_m = list()
  for new_row in zip(*m):  # zip(*m) is m transposed
    new_m.append(list(reversed(new_row)))  # transposed gives us our rows in the reverse order than we want.
  return new_m

def rotate_matrix_in_place(m: List[List[Any]]) -> List[List[Any]]:
  """7. Rotate Matrix: Rotate an NxN matrix by 90-degrees clockwise.

  Given an image represented by an NxN matrix, where each pixel in the image is 4 bytes, write a method to rotate the
  image by 90 degrees. Can you do this in place?
  """
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
  """8. Zero Matrix: Modify an MxN matrix such that if an element is 0, its entire row and column are set to O."""
  # TODO: complexity analysis
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

def zero_matrix_in_place(m: List[List[Any]]) -> List[List[Any]]:
  """8. Zero Matrix: Modify an MxN matrix such that if an element is 0, its entire row and column are set to O."""
  # TODO: complexity analysis
  if len(m) <= 1:
    return m

  num_rows = len(m)
  num_cols = len(m[0])

  marked_rows = set()
  marked_cols = set()

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
  # TODO: complexity analysis
  return s1.find(s2) != -1

def is_substring_manual(s1: str, s2: str) -> bool:
  """Checks if s2  is a substring of s1"""
  pass  # TODO: manually implement is_substring function.

def string_rotation(s1: str, s2: str) -> bool:
  """9. String Rotation: Given two strings, s1 and s2, check if s2 is a rotation of s1.

  Assume you have a method isSubstring which checks if one word is a substring of another.
  Solve this using only one call to isSubstring (e.g., "waterbottle" is a rotation of "erbottlewat").
  """
  # TODO: complexity analysis

  if (len(s1) == 0 or len(s2) == 0) and len(s1) != len(s2):
    return False
  # if 'erbottlewat' is a rotation of 'waterbottle', then 'waterbottle' is a substring of 'erbottlewaterbottlewat'
  # in other words:
  # if s2 is a rotation of s1, then s1 is a substring of s2+s2
  return is_substring(s2+s2, s1)

def string_rotation_efficient(m: List[List[Any]]) -> List[List[Any]]:
  """9. String Rotation: Given two strings, s1 and s2, check if s2 is a rotation of s1.

  Assume you have a method isSubstring which checks if one word is a substring of another.
  Solve this using only one call to isSubstring (e.g., "waterbottle" is a rotation of "erbottlewat").
  """
  pass  # TODO: implement using string search algos


################################################
# Main
################################################
if __name__ == '__main__':
  # checks = (
  #   ((funcs),
  #     (
  #       ((inputs), expected_output),
  #       ... # more test cases
  #     )
  #   ),
  #   ... # more functions to test
  # )

  # Test cases
  checks = (
    ################################################
    # Ch1
    ################################################
    ((is_unique, is_unique_no_data_structures,),
      (
        (('',), True),
        (('a',), True),
        (('aa',), False),
        (('asd',), True),
        (('asda',), False),
      )
    ),
    ((check_permutation,),
      (
        (('',''), True),
        (('a','a'), True),
        (('aa','aa'), True),
        (('ab','ab'), True),
        (('ab','ba'), True),
        (('abc','bca'), True),
        (('aabc','baca'), True),
        (('','a'), False),
        (('a','aa'), False),
        (('ab','aba'), False),
      )
    ),
    ((urlify,urlify_manual,),
      (
        (('',), ''),
        (('a',), 'a'),
        ((' ',), '%20'),
        (('  ',), '%20%20'),
        ((' a',), '%20a'),
        (('a ',), 'a%20'),
        ((' a ',), '%20a%20'),
        (('Mr John Smith',), 'Mr%20John%20Smith'),
      )
    ),
    ((palindrome_permutation,),
      (
        (('',), True),
        (('a',), True),
        (('aa',), True),
        (('aaa',), True),
        (('aba',), True),
        (('aab',), True),
        (('abba',), True),
        (('aabb',), True),
        (('abbcccc',), True),
        (('ab',), False),
        (('abbb',), False),
        (('abc',), False),
        (('Tact Coa',), True),
      )
    ),
    ((one_away,),
      (
        (('',''), True),
        (('a','a'), True),
        (('ab','ab'), True),
        (('ab','a'), True),
        (('','a'), True),
        (('a','ab'), True),
        (('a','aaa'), False),
        (('','ab'), False),
        (('a','abc'), False),
        (('abx','abb'), True),
        (('abx','abc'), True),
        (('axx','abb'), False),
        (('pale','ple'), True),
        (('pales','pale'), True),
        (('pale','bale'), True),
        (('pale','bake'), False),
      )
    ),
    ((string_compression,),
      (
        (('',), ''),
        (('a',), 'a'),
        (('aa',), 'aa'),
        (('ab',), 'ab'),
        (('aaa',), 'a3'),
        (('aaab',), 'aaab'),
        (('aaaab',), 'a4b1'),
        (('aabbcc',), 'aabbcc'),
        (('aaabbcc',), 'a3b2c2'),
        (('aabcccccaaa',), 'a2b1c5a3'),
      )
    ),
    ((rotate_matrix,rotate_matrix_in_place,),
      (
        (([[]],), [[]]),
        (([[0]],), [[0]]),
        (([[0,0],[0,0]],), [[0,0],[0,0]]),
        (([[1,2],[3,4]],), [[3,1],[4,2]]),
        (([[1,2,3],[4,5,6],[7,8,9]],), [[7,4,1],[8,5,2],[9,6,3]]),
        (([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],), [[13,9,5,1],[14,10,6,2],[15,11,7,3],[16,12,8,4]]),
        (([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]],), [[21,16,11,6,1],[22,17,12,7,2],[23,18,13,8,3],[24,19,14,9,4],[25,20,15,10,5]]),
      )
    ),
    ((zero_matrix,zero_matrix_in_place,),
      (
        (([[]],), [[]]),
        (([[0]],), [[0]]),
        (([[1,2],[3,4]],), [[1,2],[3,4]]),
        (([[0,2],[3,4]],), [[0,0],[0,4]]),
        (([[1,0],[3,4]],), [[0,0],[3,0]]),
        (([[0,0],[3,4]],), [[0,0],[0,0]]),
        (([[1,2,3],[4,5,6],[7,8,9]],), [[1,2,3],[4,5,6],[7,8,9]]),
        (([[0,2,3],[4,5,6],[7,8,9]],), [[0,0,0],[0,5,6],[0,8,9]]),
        (([[1,0,3],[4,5,6],[7,8,9]],), [[0,0,0],[4,0,6],[7,0,9]]),
        (([[1,2,3],[4,0,6],[7,8,9]],), [[1,0,3],[0,0,0],[7,0,9]]),
        (([[0,2,3],[4,5,6],[7,8,0]],), [[0,0,0],[0,5,0],[0,0,0]]),
      )
    ),
    ((is_substring,),
      (
        (('',''), True),
        (('a',''), True),
        (('ab',''), True),
        (('','a'), False),
        (('a','a'), True),
        (('ab','a'), True),
        (('ba','a'), True),
        (('ab','ab'), True),
        (('ab','ba'), False),
        (('ba','ab'), False),
        (('abb','bb'), True),
        (('bab','bb'), False),
        (('bba','bb'), True),
        (('erbottlewaterbottlewat','waterbottle'), True),
      )
    ),
    ((string_rotation,),
      (
        (('',''), True),
        (('a',''), False),
        (('ab',''), False),
        (('','a'), False),
        (('a','a'), True),
        (('ab','a'), False),
        (('ba','a'), False),
        (('ab','ab'), True),
        (('ab','ba'), True),
        (('ba','ab'), True),
        (('abb','bb'), False),
        (('bab','bb'), False),
        (('bba','bb'), False),
        (('waterbottle','erbottlewat'), True),
      )
    ),


    ################################################
    # Ch2
    ################################################
  )

  ################################################
  # Run tests and output results
  ################################################
  # Log output flags -- TODO: change to actual flags system. TODO: Add optional commandline parameterization.
  PRINT_FUNCTIONS = True
  PRINT_FUNCTION_ANNOTATIONS = False
  PRINT_TESTS = False
  PRINT_FAILS = True

  # Init
  fails = dict()
  func_count = test_count = 0
  func_fail_count = test_fail_count = 0

  # Loop through each check
  for check_num, (funcs, tests) in enumerate(checks, 1):
    # Loop through each function in current check
    for func_num, func_ in enumerate(funcs, 1):
      func_count += 1

      # Function output
      if PRINT_FUNCTIONS or PRINT_TESTS or PRINT_FAILS:
        top_level_log_output = f'Check #{check_num:02}.{func_num:02}: {func_.__name__}{" "+str(func_.__annotations__) if PRINT_FUNCTION_ANNOTATIONS else ""}:'
        if PRINT_FUNCTIONS or PRINT_TESTS:
          print(top_level_log_output[:None if PRINT_TESTS else -1])  # Do not print trailing ':' if not PRINT_TESTS

      # Loop through each test case in current check
      #   -- note: a check can have multiple functions, but all of these functions will receive the same tests
      for test_case_num, (test_input_tuple, expected_output) in enumerate(tests, 1):
        test_count += 1
        actual_output = func_(*test_input_tuple)
        test_succeeded = actual_output == expected_output

        # Test case output
        if PRINT_TESTS or (PRINT_FAILS and not test_succeeded):
          log_output = \
            f'    Test case #{test_case_num:02}: {"PASSED" if actual_output == expected_output else "FAILED <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"}\n' \
            f'      Input:            {test_input_tuple}\n' \
            f'      Expected output:  {expected_output}\n' \
            f'      Actual output:    {actual_output}\n'
          if PRINT_TESTS:
            print(log_output)

        # Tracking failed test cases 
        if not test_succeeded:
          if (check_num, func_num) in fails:
            test_fail_count += 1
            if PRINT_FAILS:
              fails[(check_num, func_num)][1].append(log_output)
          else:
            func_fail_count += 1
            test_fail_count += 1
            if PRINT_FAILS:
              fails[(check_num, func_num)] = [top_level_log_output, [log_output]]
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

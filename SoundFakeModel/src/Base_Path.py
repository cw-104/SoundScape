import os

def get_path_relative_base(path):
  """
  Converts a relative path to an absolute path, starting from the base layer of the project.
  """
  

  base_dir = os.path.dirname(os.path.abspath(__file__))

 
  absolute_path = os.path.join(base_dir, path)

  return absolute_path

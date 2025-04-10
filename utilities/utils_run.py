"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import subprocess


def run_command_sync(command, debug=False):
  """ runs command and returns the output."""
  if debug:
      print("$ {}".format(command))
  p = subprocess.Popen(command, shell=True, executable='/bin/bash')
  return p 
    

def run_command_async(command, output_log_filename=None):
  # Redirect the standard output and standard error output if specifying an output file
  redirected_std_out = subprocess.PIPE
  redirected_std_err = subprocess.PIPE
  if output_log_filename:
      output_log_file = open(output_log_filename, "wb")
      redirected_std_out = output_log_file
      redirected_std_err = output_log_file  
  process = subprocess.Popen(
      command,
      shell = True,
      stdout = redirected_std_out,
      stderr = redirected_std_err,
      text = True  # Use text=True to get text output
  )
  stdout, stderr = process.communicate()
  return process.returncode, stdout, stderr
  
#!/usr/bin/env python3
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

import os
import argparse
from pathlib import Path
import numpy as np 
import subprocess
import platform

can_convert_html_to_pdf = True 
try:
    import pdfkit
    from pypdf import PdfReader
except ImportError as e:
    can_convert_html_to_pdf = False 
    print('run on you shell:\n pip install pdfkit \n brew install Caskroom/cask/wkhtmltopdf \n pip install pypdf')
    print('Under Ubuntu: sudo apt-get install wkhtmltopdf')

script_dir = Path(os.path.realpath(__file__)).parent 
arcturus_dir = script_dir.parent


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.description = "Plot tracker information"
    
    parser.add_argument('-f','--file', help=("A list of results.json files"))
    parser.add_argument('-w','--width', help=("Image width in inches"), default=14)
    
    # parse cmdline args
    args = vars(parser.parse_args())
    return args  


def main():
    """Main."""

    parser_args = parse_command_line()
    print(parser_args)
    
    input_filename = parser_args['file']
    image_width_inches = parser_args['width']
    
    path = os.path.dirname(input_filename)
    filename = os.path.basename(input_filename)
    filename_no_ext = os.path.splitext(filename)[0]
    
    pdf_report_path = path + '/' + filename_no_ext + '.pdf'
    
    if can_convert_html_to_pdf:
        #  for pdfkit.from_url options, see https://pypi.python.org/pypi/pdfkit
        
        # first, print fake to get the full height of the document 
        options = {
            'dpi':200,
            'enable-local-file-access': None, 
            'page-width':  f'{image_width_inches}in',
            'page-height': f'{image_width_inches}in'
        }
        pdfkit.from_url(input_filename, pdf_report_path, options)

        # read the document to get the final size (from https://stackoverflow.com/questions/6230752/extracting-page-sizes-from-pdf-in-python)
        reader = PdfReader(pdf_report_path)
        box = reader.pages[0].mediabox        
        width = box.width * 1/72
        height = box.height * 1/72
        print(f'width: {width}, height: {height}')
        print('pages: ', len(reader.pages))
        
        total_height = height * len(reader.pages)
                
        # then, print the final version (multipage) by using the previous computed total height information 
        options = {
            'dpi':200,
            'enable-local-file-access': None, 
            'page-width':  f'{image_width_inches}in',
            'page-height': f'{total_height}in'
        }
        pdfkit.from_url(parser_args['file'], pdf_report_path, options)
        print('Saved pdf report to: ' + pdf_report_path)
        if platform.system() == "Darwin":
            subprocess.call(["open", pdf_report_path])
        else: 
            subprocess.call(["xdg-open", pdf_report_path])
            
if __name__ == "__main__":
    main()

import os

"""
We write all of our tutorials in jupyter notebooks (b/c it handle images and plots natively and is easier),
and then convert them to restructured text. Run this file to do the conversion and put everything in it's 
correct place. The only drawback is if you want to put  inline images in, put the image in a folder 
named {file}_files.

Requires that nbconvert is installed on your system as well.
"""

def convert_jupyter(file):
    #do initial conversions
    os.system(f'jupyter nbconvert --to rst {file}')

    #remove previous and move data to static
    filename = file.split(".")[0]
    files = fr'{filename}_files'
    os.system(f'rm -rf ../../docs/source/tutorials/{files}')
    os.system(f'cp -r {files} ../../docs/source/tutorials')
    os.system(f'rm {files}/*_0.png')
    os.system(f'mv {filename}.rst ../../docs/source/tutorials/{filename}.rst')

def convert_all():
    #convert all posts
    for filename in os.listdir("./"):
        if filename.endswith(".ipynb"):
            convert_jupyter(filename)

if __name__ == "__main__":
    convert_all()
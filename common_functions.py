import re

def clean_name(row, remove, columnName):
    name_string = row[columnName]
    if isinstance(name_string,str):
        cleaned_name = re.sub(remove, '', name_string)
        return cleaned_name
    else:
        return ''
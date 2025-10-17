#!/usr/bin/python
import os 
# paths = r"./ModelNet40"
paths = os.walk(r"../ModelNet40")

for path, dir_lst, file_lst in paths:
    for file_name   in file_lst:
        # print(os.path.join(path, f_name))
        f_name = os.path.join(path, file_name)
        print("> Reading file: " + f_name)
        in_file = open(f_name, "r")
        all_lines = in_file.readlines()
        first_line = all_lines[0]
        tokens = first_line.split()
        # should have only one token: OFF
        if(len(tokens) > 1):
            # Get the value with the 'OFF'
            tokens[0] = tokens[0].split("OFF")[1]
            edited_line = tokens[0] + " " + tokens[1] + " " + tokens[2] + "\n"
            all_lines[0] = edited_line
            all_lines = ["OFF\n"] + all_lines
            in_file.close()
            in_file = open(f_name, "w")
            in_file.writelines(all_lines)
            in_file.close()
        else:
            in_file.close()
        
        print('> Done!')
print('ModelNet40 Fixer Done!')


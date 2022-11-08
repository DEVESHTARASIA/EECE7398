import sys
import argparse

print("No. of arguments passed: ",len(sys.argv))
print("First argument is the name of the script: ",sys.argv[0])
print("Can this be a string? ",type(sys.argv[1]))
print("Is it an integer",type(sys.argv[2])) #Everything is understood as string
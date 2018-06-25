#!/bin/sh

#  Junk_Code.sh
#  
#
#  Created by Michael Varley on 22/06/2018.
#
"germancredit.intervals.data"

testfile=open("germancredit.ts.data","r")
original_data=testfile.read().split('\n')
original_array=[]
for j in original_data:
k=j.split(',')
original_array.append(k)

intervalfile=open("germancredit.intervals.data","r")
original_intervals=testfile.read.split('\n')
intervals=[]
for l in original_intervals:
m=j.split(',')
intervals.append(m)

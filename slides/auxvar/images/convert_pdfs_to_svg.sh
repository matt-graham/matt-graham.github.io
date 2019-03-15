#!/bin/bash
for filename in *.pdf; do
  pdf2svg "$filename" "${filename%.*}.svg"
done

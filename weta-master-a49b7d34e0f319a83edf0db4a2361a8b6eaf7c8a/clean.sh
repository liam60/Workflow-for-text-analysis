#!/bin/bash

#rm ~/Library/Caches/Orange/3.4.5/canvas/widget-registry.pck
#rm ~/Library/Caches/Orange/3.4.5/canvas/registry-cache.pck

rm -rf ~/Library/Caches/Orange

pip uninstall weta
pip install -e .
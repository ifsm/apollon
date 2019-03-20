FLAGS=-fno-common -dynamic -DNDEBUG -g -fwrapv -O3 -arch x86_64
WARNINGS=-Wno-unused-result -Wsign-compare -Wunreachable-code -Wall
INCLUDE_PYTHON=-I/Library/Frameworks/Python.framework/Versions/3.7/include/python3.7m/
INCLUDE_NUMPY=-I/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/numpy/core/include/

pf: psycho_features_module.c
	gcc -c $(FLAGS) $(WARNINGS) $(INCLUDE_PYTHON) $(INCLUDE_NUMPY) psycho_features_module.c -o psycho_features.so

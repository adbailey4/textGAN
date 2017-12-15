all : python-recs python_utils BaseTensor textGAN

python-recs: 
	pip install tensorflow==1.4.0

BaseTensor:
	cd BaseTensor && python setup.py install

textGAN:
	python setup.py install

python_utils:
	cd python_utils && python setup.py install

test:
	pytest

.PHONY: BaseTensor textGAN python-recs python_utils

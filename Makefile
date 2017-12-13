all : python-recs BaseTensor textGAN

python-recs: 
	pip install tensorflow==1.4.0

BaseTensor:
	cd BaseTensor && python setup.py install

textGAN:
	python setup.py install

test:
	pytest

.PHONY: BaseTensor textGAN python-recs

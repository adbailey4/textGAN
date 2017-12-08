all : BaseTensor textGAN

BaseTensor:
	cd BaseTensor && python setup.py install

textGAN:
	python setup.py install

test:
	pytest

.PHONY: BaseTensor textGAN

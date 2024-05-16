
all: device_diagnostic

device_diagnostic: device_diagnostic.cu
	nvcc -o $@ $<

clean:
	rm -f device_diagnostic

.PHONY: clean all

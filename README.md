# nPort

Python code for dealing with 2-port and higher representations of networks, such as scattering
parameters, impedance parameters and so forth. For applications in electric circuits,
or for characterising the reflection and transmission of microwave, optical or terahertz samples
in experiment or simulation.

## Major features

- Loads files in Touchstone file format, which is commonly used by vector network analysers,
  and electromagnetic simulation software.

- Converts between different parameters (See the definitions on the [Wikipedia page](http://en.wikipedia.org/wiki/Two-port_network))
    - admittance (Y), arbitrary number of ports
    - scattering (S), arbitrary number of ports
    - impedance (Z), arbitrary number of ports
    - transfer (ABCD), two ports only
    - scattering transfer (T), two ports only
    Not every conversion is implemented

- A few other miscellaneous conversions, such as between linear and circular polarisation
    
## Installation

Requires `numpy`, and to plot the results a plotting library such as `matplotlib` is recommended.

`pip install https://github.com/davidpowell/nport/archive/master.zip`

## Example

Load and plot scattering parameters from a file

```python
from nport import loadsnp
import matplotlib.pyplot as plt

S = loadsnp('my_parameters.s2p')

plt.figure()
plt.plot(S.f, abs(S[2,1]))
plt.show()
```


[comment]: <> (markdown_py -x markdown.extensions.fenced_code < README.md > README.html)


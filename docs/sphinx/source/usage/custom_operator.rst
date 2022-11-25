Defining a Custom Operator
==============================


Operator Interface
------------------

In finch, an operator can have multiple different implementations.
However, every implementation of the same operator must support a common interface.
An implementation of an operator is in principle just a function.
In principle, any combinations of in- and outputs are allowed.
However, finch has a default interface which is assumed when running experiments.
An operator conforming to the default interface takes a single ``xarray.Dataset`` as input and returns a ``xarray.DataArray`` as output.

Defining Inputs
------------------
In order to seamlessly run an operator implementation on data, finch provides its own input management via the :class:`finch.Input` class.
An object of this class wraps some source of data, which generates an input for the operator.

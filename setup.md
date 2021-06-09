---
layout: page
title: Setup
permalink: /setup/
---

This lesson will be carried out on our virtual cluster.

~~~
ssh userXX@pcs.ace-net.training
~~~
{: .bash}

It should be equally possible to use one of the Compute Canada
general purpose clusters if you already have an account there.
To set up your environment, load the NVidia HPC Software Development Kit with:

~~~
module load StdEnv/2020
module load nvhpc
~~~
{: .bash}

You should have access to the compiler and tools now. Test this with:

~~~
which nvcc
which nvprof
~~~
{: .bash}

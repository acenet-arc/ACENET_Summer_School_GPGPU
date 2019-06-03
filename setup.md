---
layout: page
title: Setup
permalink: /setup/
---

This lesson will be carried out on the Béluga cluster.

~~~
ssh username@beluga.computecanada.ca
~~~
{: .bash}

To set up your environment, load the appropriate module with:

~~~
module purge
module load cuda
~~~
{: .bash}

You should have access to the compiler and tools now. Test this with:

~~~
which nvcc
which nvprof
~~~
{: .bash}

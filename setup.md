---
layout: page
title: Setup
permalink: /setup/
---

This lesson will be carried out on our virtual cluster.

~~~
ssh userXX@nova.acenetsummerschool.ca
~~~
{: .bash}

To set up your environment, load the appropriate module with:

~~~
module load cuda
~~~
{: .bash}

You should have access to the compiler and tools now. Test this with:

~~~
which nvcc
which nvprof
~~~
{: .bash}
